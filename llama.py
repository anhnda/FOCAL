from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.nn.functional import softmax

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
MODEL_NAME = "meta-llama/Llama-3.2-1B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

# ----------------------------------------------------------------------
# Load pretrained Llama-3.2-1B
# ----------------------------------------------------------------------
# NOTE: Llama models are gated on the Hugging Face Hub. You must:
#   1. Request access at https://huggingface.co/meta-llama/Llama-3.2-1B
#   2. Authenticate locally, e.g.  `huggingface-cli login`
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=DTYPE,
)
model.to(DEVICE)
model.eval()

# Llama tokenizers usually have no pad token defined -> reuse the EOS token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Prompting...")

# ----------------------------------------------------------------------
# Prompt
# ----------------------------------------------------------------------
prompt = "In the future, artificial intelligence will"

input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)

# ----------------------------------------------------------------------
# Text generation with different sampling settings
# ----------------------------------------------------------------------
def generate_text(temperature=1.0, top_k=0, top_p=1.0):
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=50,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)


settings = [
    {"temperature": 0.7, "top_k": 0,  "top_p": 1.0},
    {"temperature": 1.0, "top_k": 50, "top_p": 1.0},
    {"temperature": 1.0, "top_k": 0,  "top_p": 0.9},
    {"temperature": 1.5, "top_k": 40, "top_p": 0.9},
]

for i, s in enumerate(settings):
    print(f"\n--- Sample {i + 1} ---")
    print(f"Temp={s['temperature']} | Top-k={s['top_k']} | Top-p={s['top_p']}")
    print(generate_text(**s))

# ----------------------------------------------------------------------
# Token-level predictive uncertainty
# ----------------------------------------------------------------------
def get_token_probs(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0]          # shape: [seq_len, vocab_size]
        probs = softmax(logits, dim=-1)
    token_probs = probs.max(dim=-1).values.cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    return tokens, token_probs


tokens, token_probs = get_token_probs(prompt)

# Clean up the tokenizer's space marker (Llama uses the SentencePiece-style "Ġ"
# in some configs or a leading "▁"); strip it so the labels are readable.
clean_tokens = [t.replace("Ġ", " ").replace("▁", " ").strip() or t for t in tokens]

plt.figure(figsize=(12, 4))
sns.barplot(x=clean_tokens, y=1 - token_probs)  # 1 - max prob = uncertainty
plt.xticks(rotation=45)
plt.title("Token-Level Predictive Uncertainty (1 - max prob)")
plt.ylabel("Uncertainty")
plt.tight_layout()
plt.show()