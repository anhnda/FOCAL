
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# Load pretrained GPT-2
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

if torch.cuda.is_available():
    model.cuda()
print("Promting...")
# Prompt
prompt = "In the future, artificial intelligence will"

# Encode input
input_ids = tokenizer.encode(prompt, return_tensors="pt")
if torch.cuda.is_available():
    input_ids = input_ids.cuda()

###


def generate_text(temperature=1.0, top_k=0, top_p=1.0):
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=50,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)



settings = [
    {"temperature": 0.7, "top_k": 0, "top_p": 1.0},
    {"temperature": 1.0, "top_k": 50, "top_p": 1.0},
    {"temperature": 1.0, "top_k": 0, "top_p": 0.9},
    {"temperature": 1.5, "top_k": 40, "top_p": 0.9},
]

for i, s in enumerate(settings):
    print(f"\n--- Sample {i+1} ---")
    print(f"Temp={s['temperature']} | Top-k={s['top_k']} | Top-p={s['top_p']}")
    print(generate_text(**s))


###

from torch.nn.functional import softmax

# Get logits for each next token
def get_token_probs(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0]  # shape: [seq_len, vocab_size]
        probs = softmax(logits, dim=-1)

    token_probs = probs.max(dim=-1).values.cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    return tokens, token_probs

tokens, token_probs = get_token_probs(prompt)

plt.figure(figsize=(12, 4))
sns.barplot(x=tokens, y=1 - token_probs)  # 1 - max prob = token-level uncertainty
plt.xticks(rotation=45)
plt.title("Token-Level Predictive Uncertainty (1 - max prob)")
plt.ylabel("Uncertainty")
plt.tight_layout()
plt.show()
