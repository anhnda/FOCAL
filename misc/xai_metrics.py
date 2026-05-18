"""
XAI Metrics for Attribution Evaluation

This module contains metrics for evaluating token-level attributions:
- Log-odds: Change in log probability when important tokens are masked
- Comprehensiveness: Probability drop when important tokens are removed
- Sufficiency: How well top-k tokens alone preserve the prediction

Includes both classification metrics and QA-specific metrics.
"""

import numpy as np
import torch
import torch.nn.functional as F

# CLASSIFICATION

def calculate_log_odds(foward_func, model, input_embed, position_embed, type_embed, attention_mask, base_token_emb, attr, topk=20):
	"""Calculate log-odds for classification tasks."""
	logits_original						= foward_func(model, input_embed, attention_mask=attention_mask, position_embed=position_embed, type_embed=type_embed, return_all_logits=True).squeeze()
	predicted_label						= torch.argmax(logits_original).item()
	prob_original						= torch.softmax(logits_original, dim=0)
	topk_indices						= torch.topk(attr, int(attr.shape[0] * topk / 100), sorted=False).indices
	local_input_embed					= input_embed.detach().clone()
	local_input_embed[0][topk_indices]	= base_token_emb
	logits_perturbed					= foward_func(model, local_input_embed, attention_mask=attention_mask, position_embed=position_embed, type_embed=type_embed, return_all_logits=True).squeeze()
	prob_perturbed						= torch.softmax(logits_perturbed, dim=0)

	return (torch.log(prob_perturbed[predicted_label]) - torch.log(prob_original[predicted_label])).item(), predicted_label


def calculate_sufficiency(foward_func, model, input_embed, position_embed, type_embed, attention_mask, base_token_emb, attr, topk=20):
	"""Calculate sufficiency for classification tasks."""
	logits_original							= foward_func(model, input_embed, attention_mask=attention_mask, position_embed=position_embed, type_embed=type_embed, return_all_logits=True).squeeze()
	predicted_label							= torch.argmax(logits_original).item()
	prob_original							= torch.softmax(logits_original, dim=0)
	topk_indices							= torch.topk(attr, int(attr.shape[0] * topk / 100), sorted=False).indices
	if len(topk_indices) == 0:
		# topk% is too less to select even word - so no masking will happen.
		return 0

	mask									= torch.zeros_like(input_embed[0][:,0]).bool()
	mask[topk_indices]						= 1
	masked_input_embed						= input_embed[0][mask].unsqueeze(0)
	masked_attention_mask					= None if attention_mask is None else attention_mask[0][mask].unsqueeze(0)
	masked_position_embed					= None if position_embed is None else position_embed[0][:mask.sum().item()].unsqueeze(0)
	masked_type_embed						= None if type_embed is None else type_embed[0][mask].unsqueeze(0)
	logits_perturbed						= foward_func(model, masked_input_embed, attention_mask=masked_attention_mask, position_embed=masked_position_embed, type_embed=masked_type_embed, return_all_logits=True).squeeze()
	prob_perturbed							= torch.softmax(logits_perturbed, dim=0)

	return (prob_original[predicted_label] - prob_perturbed[predicted_label]).item()


def calculate_comprehensiveness(foward_func, model, input_embed, position_embed, type_embed, attention_mask, base_token_emb, attr, topk=20):
	"""Calculate comprehensiveness for classification tasks."""
	logits_original					= foward_func(model, input_embed, attention_mask=attention_mask, position_embed=position_embed, type_embed=type_embed, return_all_logits=True).squeeze()
	predicted_label					= torch.argmax(logits_original).item()
	prob_original					= torch.softmax(logits_original, dim=0)
	topk_indices					= torch.topk(attr, int(attr.shape[0] * topk / 100), sorted=False).indices
	mask 							= torch.ones_like(input_embed[0][:,0]).bool()
	mask[topk_indices]				= 0
	masked_input_embed				= input_embed[0][mask].unsqueeze(0)
	masked_attention_mask			= None if attention_mask is None else attention_mask[0][mask].unsqueeze(0)
	masked_position_embed			= None if position_embed is None else position_embed[0][:mask.sum().item()].unsqueeze(0)
	masked_type_embed				= None if type_embed is None else type_embed[0][mask].unsqueeze(0)
	logits_perturbed				= foward_func(model, masked_input_embed, attention_mask=masked_attention_mask, position_embed=masked_position_embed, type_embed=masked_type_embed, return_all_logits=True).squeeze()
	prob_perturbed					= torch.softmax(logits_perturbed, dim=0)

	return (prob_original[predicted_label] - prob_perturbed[predicted_label]).item()


# QUESTION ANSWERING

def _to_tensor_scalar(x, device):
    """Ensure x is a 0-dim float Tensor on the correct device (accepts Tensor or float)."""
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=torch.float32)
    return torch.tensor(float(x), dtype=torch.float32, device=device)


def calculate_log_odds_qa(model, input_embed, attention_mask, special_token_mask, token_type_ids, 
                          base_token_emb, attr_start, attr_end, start_idx, end_idx,
                          prob_start_orig, prob_end_orig, topk=50):
    """
    Calculate log-odds metric for Question Answering.
    
    Measures the change in log probability when top-k attributed tokens are masked.
    Uses SEPARATE attributions for start and end positions - no combined scores.
    More negative values indicate that important tokens were correctly identified.
    
    Args:
        model: QA model (AutoModelForQuestionAnswering)
        input_embed: Original input embeddings (1, L, d)
        attention_mask: Attention mask (1, L)
        special_token_mask: Boolean mask where True = special token (CLS, SEP, PAD) that should not be masked
        token_type_ids: Token type IDs (1, L) or None
        base_token_emb: Baseline token embedding for masking (1, d)
        attr_start: Attribution scores for start logit (L,) - used to select tokens for start metric
        attr_end: Attribution scores for end logit (L,) - used to select tokens for end metric
        start_idx: Predicted answer start index
        end_idx: Predicted answer end index
        prob_start_orig: Original probability of predicted start index (Tensor or float)
        prob_end_orig: Original probability of predicted end index (Tensor or float)
        topk: Percentage of top tokens to mask (default: 50%)
    
    Returns:
        Tuple of (log_odds_start, log_odds_end): Log odds difference for start and end positions
    """
    device = input_embed.device
    # Normalise scalar probs so torch.log always receives a Tensor
    prob_start_orig = _to_tensor_scalar(prob_start_orig, device)
    prob_end_orig   = _to_tensor_scalar(prob_end_orig,   device)

    extra_kwargs = {}
    if token_type_ids is not None:
        extra_kwargs["token_type_ids"] = token_type_ids
     
    num_tokens = attr_start.shape[0]
    # Count maskable tokens (excluding special tokens)
    num_maskable = num_tokens - special_token_mask.sum().item()
    k = max(1, int(num_maskable * topk / 100))
    
    # ===== Compute log_odds for START using attr_start =====
    attr_start_masked = attr_start.clone()
    attr_start_masked[special_token_mask] = float('-inf')
    topk_indices_start = torch.topk(attr_start_masked, k, sorted=False).indices
    
    perturbed_embed_start = input_embed.detach().clone()
    perturbed_embed_start[0][topk_indices_start] = base_token_emb
    
    with torch.no_grad():
        outputs_pert_start = model(inputs_embeds=perturbed_embed_start, attention_mask=attention_mask, **extra_kwargs)
        start_logits_pert = outputs_pert_start.start_logits[0]
        prob_start_pert = F.softmax(start_logits_pert, dim=0)[start_idx]
    
    log_odds_start = (torch.log(prob_start_pert + 1e-10) - torch.log(prob_start_orig + 1e-10)).item()
    
    # ===== Compute log_odds for END using attr_end =====
    attr_end_masked = attr_end.clone()
    attr_end_masked[special_token_mask] = float('-inf')
    topk_indices_end = torch.topk(attr_end_masked, k, sorted=False).indices
    
    perturbed_embed_end = input_embed.detach().clone()
    perturbed_embed_end[0][topk_indices_end] = base_token_emb
    
    with torch.no_grad():
        outputs_pert_end = model(inputs_embeds=perturbed_embed_end, attention_mask=attention_mask, **extra_kwargs)
        end_logits_pert = outputs_pert_end.end_logits[0]
        prob_end_pert = F.softmax(end_logits_pert, dim=0)[end_idx]
    
    log_odds_end = (torch.log(prob_end_pert + 1e-10) - torch.log(prob_end_orig + 1e-10)).item()
    
    return log_odds_start, log_odds_end


def calculate_comprehensiveness_qa(model, input_embed, attention_mask, special_token_mask, token_type_ids,
                                   base_token_emb, attr_start, attr_end, start_idx, end_idx,
                                   prob_start_orig, prob_end_orig, topk=50):
    """
    Calculate comprehensiveness metric for Question Answering.
    
    Measures probability drop when top-k attributed tokens are removed.
    Uses SEPARATE attributions for start and end positions - no combined scores.
    Higher values indicate that important tokens were correctly identified.
    
    Args:
        model: QA model
        input_embed: Original input embeddings (1, L, d)
        attention_mask: Attention mask (1, L)
        special_token_mask: Boolean mask where True = special token (CLS, SEP, PAD) that should not be removed
        token_type_ids: Token type IDs (1, L) or None
        base_token_emb: Baseline token embedding (1, d)
        attr_start: Attribution scores for start logit (L,) - used to select tokens for start metric
        attr_end: Attribution scores for end logit (L,) - used to select tokens for end metric
        start_idx: Predicted answer start index
        end_idx: Predicted answer end index
        prob_start_orig: Original probability of predicted start index (Tensor or float)
        prob_end_orig: Original probability of predicted end index (Tensor or float)
        topk: Percentage of top tokens to remove (default: 50%)
    
    Returns:
        Tuple of (comp_start, comp_end): Probability drop for start and end positions
    """
    device = input_embed.device
    # Normalise scalar probs
    prob_start_orig = _to_tensor_scalar(prob_start_orig, device)
    prob_end_orig   = _to_tensor_scalar(prob_end_orig,   device)

    extra_kwargs = {}
    if token_type_ids is not None:
        extra_kwargs["token_type_ids"] = token_type_ids
    
    # ===== Compute comprehensiveness for START using attr_start =====
    attr_start_masked = attr_start.clone()
    attr_start_masked[special_token_mask] = float('-inf')
    topk_indices_start = torch.topk(attr_start_masked, int(attr_start.shape[0] * topk / 100), sorted=False).indices
    
    perturbed_embed_start = input_embed.detach().clone()
    perturbed_embed_start[0][topk_indices_start] = base_token_emb
    
    with torch.no_grad():
        outputs_pert_start = model(inputs_embeds=perturbed_embed_start, attention_mask=attention_mask, **extra_kwargs)
        start_logits_pert = outputs_pert_start.start_logits[0]
        new_len_start = start_logits_pert.shape[0]
        new_start_idx = min(start_idx, new_len_start - 1)
        prob_start_pert = F.softmax(start_logits_pert, dim=0)[new_start_idx]
    
    comp_start = (prob_start_orig - prob_start_pert).item()
    
    # ===== Compute comprehensiveness for END using attr_end =====
    attr_end_masked = attr_end.clone()
    attr_end_masked[special_token_mask] = float('-inf')
    topk_indices_end = torch.topk(attr_end_masked, int(attr_end.shape[0] * topk / 100), sorted=False).indices
    
    perturbed_embed_end = input_embed.detach().clone()
    perturbed_embed_end[0][topk_indices_end] = base_token_emb
    
    with torch.no_grad():
        outputs_pert_end = model(inputs_embeds=perturbed_embed_end, attention_mask=attention_mask, **extra_kwargs)
        end_logits_pert = outputs_pert_end.end_logits[0]
        new_len_end = end_logits_pert.shape[0]
        new_end_idx = min(end_idx, new_len_end - 1)
        prob_end_pert = F.softmax(end_logits_pert, dim=0)[new_end_idx]
    
    comp_end = (prob_end_orig - prob_end_pert).item()
    
    return comp_start, comp_end


def calculate_sufficiency_qa(model, input_embed, attention_mask, special_token_mask, token_type_ids,
                             base_token_emb, attr_start, attr_end, start_idx, end_idx,
                             prob_start_orig, prob_end_orig, topk=50):
    """
    Calculate sufficiency metric for Question Answering (MASKING, not deletion).

    Keeps only the top-k attributed tokens (plus special tokens) by masking all other tokens
    with base_token_emb while keeping sequence length fixed.

    Args:
        prob_start_orig: Original probability of predicted start index (Tensor or float)
        prob_end_orig: Original probability of predicted end index (Tensor or float)

    Returns:
        Tuple of (suff_start, suff_end): Probability drop for start and end positions
    """
    device = input_embed.device
    # Normalise scalar probs
    prob_start_orig = _to_tensor_scalar(prob_start_orig, device)
    prob_end_orig   = _to_tensor_scalar(prob_end_orig,   device)

    extra_kwargs = {}
    if token_type_ids is not None:
        extra_kwargs["token_type_ids"] = token_type_ids

    num_tokens = attr_start.shape[0]
    num_maskable = num_tokens - special_token_mask.sum().item()
    k = max(1, int(num_maskable * topk / 100))

    # ===== Compute sufficiency for START using attr_start =====
    attr_start_masked = attr_start.clone()
    attr_start_masked[special_token_mask] = float('-inf')
    topk_indices_start = torch.topk(attr_start_masked, int(attr_start.shape[0] * topk / 100), sorted=False).indices

    keep_mask_start = torch.zeros(num_tokens, dtype=torch.bool, device=device)
    keep_mask_start[topk_indices_start] = True
    keep_mask_start[special_token_mask] = True

    perturbed_embed_start = input_embed.detach().clone()
    perturbed_embed_start[0, ~keep_mask_start, :] = base_token_emb

    with torch.no_grad():
        outputs_pert_start = model(inputs_embeds=perturbed_embed_start, attention_mask=attention_mask, **extra_kwargs)
        start_logits_pert = outputs_pert_start.start_logits[0]
        prob_start_pert = F.softmax(start_logits_pert, dim=0)[start_idx]

    suff_start = (prob_start_orig - prob_start_pert).item()

    # ===== Compute sufficiency for END using attr_end =====
    attr_end_masked = attr_end.clone()
    attr_end_masked[special_token_mask] = float('-inf')
    topk_indices_end = torch.topk(attr_end_masked, int(attr_end.shape[0] * topk / 100), sorted=False).indices

    keep_mask_end = torch.zeros(num_tokens, dtype=torch.bool, device=device)
    keep_mask_end[topk_indices_end] = True
    keep_mask_end[special_token_mask] = True

    perturbed_embed_end = input_embed.detach().clone()
    perturbed_embed_end[0, ~keep_mask_end, :] = base_token_emb

    with torch.no_grad():
        outputs_pert_end = model(inputs_embeds=perturbed_embed_end, attention_mask=attention_mask, **extra_kwargs)
        end_logits_pert = outputs_pert_end.end_logits[0]
        prob_end_pert = F.softmax(end_logits_pert, dim=0)[end_idx]

    suff_end = (prob_end_orig - prob_end_pert).item()

    return suff_start, suff_end


def eval_wae(scaled_features, word_embedding, epsilon=0.1):
	"""
	Compute Word Alignment Error (WAE).
	
	Measures the smallest distance of each embedding to any word embedding
	and reports the average among all words in the path for a sentence.
	"""
	dists = []
	for emb in scaled_features:
		all_dist = torch.sqrt(torch.sum((word_embedding - emb.unsqueeze(0)) ** 2, dim=1))
		dists.append(torch.min(all_dist).item())

	return np.mean(dists)
# """
# XAI Metrics for Attribution Evaluation

# This module contains metrics for evaluating token-level attributions:
# - Log-odds: Change in log probability when important tokens are masked
# - Comprehensiveness: Probability drop when important tokens are removed
# - Sufficiency: How well top-k tokens alone preserve the prediction

# Includes both classification metrics and QA-specific metrics.
# """

# import numpy as np
# import torch
# import torch.nn.functional as F

# # CLASSIFICATION

# def calculate_log_odds(foward_func, model, input_embed, position_embed, type_embed, attention_mask, base_token_emb, attr, topk=20):
# 	"""Calculate log-odds for classification tasks."""
# 	logits_original						= foward_func(model, input_embed, attention_mask=attention_mask, position_embed=position_embed, type_embed=type_embed, return_all_logits=True).squeeze()
# 	predicted_label						= torch.argmax(logits_original).item()
# 	prob_original						= torch.softmax(logits_original, dim=0)
# 	topk_indices						= torch.topk(attr, int(attr.shape[0] * topk / 100), sorted=False).indices
# 	local_input_embed					= input_embed.detach().clone()
# 	local_input_embed[0][topk_indices]	= base_token_emb
# 	logits_perturbed					= foward_func(model, local_input_embed, attention_mask=attention_mask, position_embed=position_embed, type_embed=type_embed, return_all_logits=True).squeeze()
# 	prob_perturbed						= torch.softmax(logits_perturbed, dim=0)

# 	return (torch.log(prob_perturbed[predicted_label]) - torch.log(prob_original[predicted_label])).item(), predicted_label


# def calculate_sufficiency(foward_func, model, input_embed, position_embed, type_embed, attention_mask, base_token_emb, attr, topk=20):
# 	"""Calculate sufficiency for classification tasks."""
# 	logits_original							= foward_func(model, input_embed, attention_mask=attention_mask, position_embed=position_embed, type_embed=type_embed, return_all_logits=True).squeeze()
# 	predicted_label							= torch.argmax(logits_original).item()
# 	prob_original							= torch.softmax(logits_original, dim=0)
# 	topk_indices							= torch.topk(attr, int(attr.shape[0] * topk / 100), sorted=False).indices
# 	if len(topk_indices) == 0:
# 		# topk% is too less to select even word - so no masking will happen.
# 		return 0

# 	mask									= torch.zeros_like(input_embed[0][:,0]).bool()
# 	mask[topk_indices]						= 1
# 	masked_input_embed						= input_embed[0][mask].unsqueeze(0)
# 	masked_attention_mask					= None if attention_mask is None else attention_mask[0][mask].unsqueeze(0)
# 	masked_position_embed					= None if position_embed is None else position_embed[0][:mask.sum().item()].unsqueeze(0)
# 	masked_type_embed						= None if type_embed is None else type_embed[0][mask].unsqueeze(0)
# 	logits_perturbed						= foward_func(model, masked_input_embed, attention_mask=masked_attention_mask, position_embed=masked_position_embed, type_embed=masked_type_embed, return_all_logits=True).squeeze()
# 	prob_perturbed							= torch.softmax(logits_perturbed, dim=0)

# 	return (prob_original[predicted_label] - prob_perturbed[predicted_label]).item()


# def calculate_comprehensiveness(foward_func, model, input_embed, position_embed, type_embed, attention_mask, base_token_emb, attr, topk=20):
# 	"""Calculate comprehensiveness for classification tasks."""
# 	logits_original					= foward_func(model, input_embed, attention_mask=attention_mask, position_embed=position_embed, type_embed=type_embed, return_all_logits=True).squeeze()
# 	predicted_label					= torch.argmax(logits_original).item()
# 	prob_original					= torch.softmax(logits_original, dim=0)
# 	topk_indices					= torch.topk(attr, int(attr.shape[0] * topk / 100), sorted=False).indices
# 	mask 							= torch.ones_like(input_embed[0][:,0]).bool()
# 	mask[topk_indices]				= 0
# 	masked_input_embed				= input_embed[0][mask].unsqueeze(0)
# 	masked_attention_mask			= None if attention_mask is None else attention_mask[0][mask].unsqueeze(0)
# 	masked_position_embed			= None if position_embed is None else position_embed[0][:mask.sum().item()].unsqueeze(0)
# 	masked_type_embed				= None if type_embed is None else type_embed[0][mask].unsqueeze(0)
# 	logits_perturbed				= foward_func(model, masked_input_embed, attention_mask=masked_attention_mask, position_embed=masked_position_embed, type_embed=masked_type_embed, return_all_logits=True).squeeze()
# 	prob_perturbed					= torch.softmax(logits_perturbed, dim=0)

# 	return (prob_original[predicted_label] - prob_perturbed[predicted_label]).item()


# # QUESTION ANSWERING

# def calculate_log_odds_qa(model, input_embed, attention_mask, special_token_mask, token_type_ids, 
#                           base_token_emb, attr_start, attr_end, start_idx, end_idx, prob_start_orig, prob_end_orig, topk=50):
#     """
#     Calculate log-odds metric for Question Answering.
    
#     Measures the change in log probability when top-k attributed tokens are masked.
#     Uses SEPARATE attributions for start and end positions - no combined scores.
#     More negative values indicate that important tokens were correctly identified.
    
#     Args:
#         model: QA model (AutoModelForQuestionAnswering)
#         input_embed: Original input embeddings (1, L, d)
#         attention_mask: Attention mask (1, L)
#         special_token_mask: Boolean mask where True = special token (CLS, SEP, PAD) that should not be masked
#         token_type_ids: Token type IDs (1, L) or None
#         base_token_emb: Baseline token embedding for masking (1, d)
#         attr_start: Attribution scores for start logit (L,) - used to select tokens for start metric
#         attr_end: Attribution scores for end logit (L,) - used to select tokens for end metric
#         start_idx: Predicted answer start index
#         end_idx: Predicted answer end index
#         prob_start_orig: Original probability of predicted start index
#         prob_end_orig: Original probability of predicted end index
#         topk: Percentage of top tokens to mask (default: 50%)
    
#     Returns:
#         Tuple of (log_odds_start, log_odds_end): Log odds difference for start and end positions
#     """
#     extra_kwargs = {}
#     if token_type_ids is not None:
#         extra_kwargs["token_type_ids"] = token_type_ids
     
#     num_tokens = attr_start.shape[0]
#     # Count maskable tokens (excluding special tokens)
#     num_maskable = num_tokens - special_token_mask.sum().item()
#     k = max(1, int(num_maskable * topk / 100))
    
#     # ===== Compute log_odds for START using attr_start =====
#     # Set special tokens to -inf so they're never selected as top-k
#     attr_start_masked = attr_start.clone()
#     attr_start_masked[special_token_mask] = float('-inf')
#     topk_indices_start = torch.topk(attr_start_masked, k, sorted=False).indices
    
#     perturbed_embed_start = input_embed.detach().clone()
#     perturbed_embed_start[0][topk_indices_start] = base_token_emb
    
#     with torch.no_grad():
#         outputs_pert_start = model(inputs_embeds=perturbed_embed_start, attention_mask=attention_mask, **extra_kwargs)
#         start_logits_pert = outputs_pert_start.start_logits[0]
#         prob_start_pert = F.softmax(start_logits_pert, dim=0)[start_idx]
    
#     log_odds_start = (torch.log(prob_start_pert + 1e-10) - torch.log(prob_start_orig + 1e-10)).item()
    
#     # ===== Compute log_odds for END using attr_end =====
#     attr_end_masked = attr_end.clone()
#     attr_end_masked[special_token_mask] = float('-inf')
#     topk_indices_end = torch.topk(attr_end_masked, k, sorted=False).indices
    
#     perturbed_embed_end = input_embed.detach().clone()
#     perturbed_embed_end[0][topk_indices_end] = base_token_emb
    
#     with torch.no_grad():
#         outputs_pert_end = model(inputs_embeds=perturbed_embed_end, attention_mask=attention_mask, **extra_kwargs)
#         end_logits_pert = outputs_pert_end.end_logits[0]
#         prob_end_pert = F.softmax(end_logits_pert, dim=0)[end_idx]
    
#     log_odds_end = (torch.log(prob_end_pert + 1e-10) - torch.log(prob_end_orig + 1e-10)).item()
    
#     return log_odds_start, log_odds_end


# def calculate_comprehensiveness_qa(model, input_embed, attention_mask, special_token_mask, token_type_ids,
#                                    base_token_emb, attr_start, attr_end, start_idx, end_idx, prob_start_orig, prob_end_orig, topk=50):
#     """
#     Calculate comprehensiveness metric for Question Answering.
    
#     Measures probability drop when top-k attributed tokens are removed.
#     Uses SEPARATE attributions for start and end positions - no combined scores.
#     Higher values indicate that important tokens were correctly identified.
    
#     Args:
#         model: QA model
#         input_embed: Original input embeddings (1, L, d)
#         attention_mask: Attention mask (1, L)
#         special_token_mask: Boolean mask where True = special token (CLS, SEP, PAD) that should not be removed
#         token_type_ids: Token type IDs (1, L) or None
#         base_token_emb: Baseline token embedding (1, d)
#         attr_start: Attribution scores for start logit (L,) - used to select tokens for start metric
#         attr_end: Attribution scores for end logit (L,) - used to select tokens for end metric
#         start_idx: Predicted answer start index
#         end_idx: Predicted answer end index
#         topk: Percentage of top tokens to remove (default: 50%)
    
#     Returns:
#         Tuple of (comp_start, comp_end): Probability drop for start and end positions
#     """
#     extra_kwargs = {}
#     if token_type_ids is not None:
#         extra_kwargs["token_type_ids"] = token_type_ids
    
#     # ===== Compute comprehensiveness for START using attr_start =====
#     # Set special tokens to -inf so they're never selected as top-k
#     attr_start_masked = attr_start.clone()
#     attr_start_masked[special_token_mask] = float('-inf')
#     topk_indices_start = torch.topk(attr_start_masked, int(attr_start.shape[0] * topk / 100), sorted=False).indices
    
#     # Create mask (keep all except top-k, always keep special tokens)
#     perturbed_embed_start = input_embed.detach().clone()
#     perturbed_embed_start[0][topk_indices_start] = base_token_emb
    
#     with torch.no_grad():
#         outputs_pert_start = model(inputs_embeds=perturbed_embed_start, attention_mask=attention_mask, **extra_kwargs)
#         start_logits_pert = outputs_pert_start.start_logits[0]
#         new_len_start = start_logits_pert.shape[0]
#         new_start_idx = min(start_idx, new_len_start - 1)
#         prob_start_pert = F.softmax(start_logits_pert, dim=0)[new_start_idx]
    
#     comp_start = (prob_start_orig - prob_start_pert).item()
    
#     # ===== Compute comprehensiveness for END using attr_end =====
#     attr_end_masked = attr_end.clone()
#     attr_end_masked[special_token_mask] = float('-inf')
#     topk_indices_end = torch.topk(attr_end_masked, int(attr_end.shape[0] * topk / 100), sorted=False).indices
    
#     perturbed_embed_end = input_embed.detach().clone()
#     perturbed_embed_end[0][topk_indices_end] = base_token_emb
    
#     extra_kwargs_end = {}
#     if token_type_ids is not None:
#         extra_kwargs_end["token_type_ids"] = token_type_ids
    
#     with torch.no_grad():
#         outputs_pert_end = model(inputs_embeds=perturbed_embed_end, attention_mask=attention_mask, **extra_kwargs_end)
#         end_logits_pert = outputs_pert_end.end_logits[0]
#         new_len_end = end_logits_pert.shape[0]
#         new_end_idx = min(end_idx, new_len_end - 1)
#         prob_end_pert = F.softmax(end_logits_pert, dim=0)[new_end_idx]
    
#     comp_end = (prob_end_orig - prob_end_pert).item()
    
#     return comp_start, comp_end


# def calculate_sufficiency_qa(model, input_embed, attention_mask, special_token_mask, token_type_ids,
#                              base_token_emb, attr_start, attr_end, start_idx, end_idx, prob_start_orig, prob_end_orig, topk=50):
#     """
#     Calculate sufficiency metric for Question Answering (MASKING, not deletion).

#     Keeps only the top-k attributed tokens (plus special tokens) by masking all other tokens
#     with base_token_emb while keeping sequence length fixed.

#     Returns:
#         Tuple of (suff_start, suff_end): Probability drop for start and end positions
#     """
#     extra_kwargs = {}
#     if token_type_ids is not None:
#         extra_kwargs["token_type_ids"] = token_type_ids

#     num_tokens = attr_start.shape[0]
#     # Count maskable tokens (excluding special tokens)
#     num_maskable = num_tokens - special_token_mask.sum().item()
#     k = max(1, int(num_maskable * topk / 100))

#     # ===== Compute sufficiency for START using attr_start =====
#     attr_start_masked = attr_start.clone()
#     attr_start_masked[special_token_mask] = float('-inf')
#     topk_indices_start = torch.topk(attr_start_masked, int(attr_start.shape[0] * topk / 100), sorted=False).indices

#     # Keep top-k + special tokens, mask the rest
#     keep_mask_start = torch.zeros(num_tokens, dtype=torch.bool, device=input_embed.device)
#     keep_mask_start[topk_indices_start] = True
#     keep_mask_start[special_token_mask] = True

#     perturbed_embed_start = input_embed.detach().clone()
#     perturbed_embed_start[0, ~keep_mask_start, :] = base_token_emb

#     with torch.no_grad():
#         outputs_pert_start = model(inputs_embeds=perturbed_embed_start, attention_mask=attention_mask, **extra_kwargs)
#         start_logits_pert = outputs_pert_start.start_logits[0]
#         prob_start_pert = F.softmax(start_logits_pert, dim=0)[start_idx]

#     suff_start = (prob_start_orig - prob_start_pert).item()

#     # ===== Compute sufficiency for END using attr_end =====
#     attr_end_masked = attr_end.clone()
#     attr_end_masked[special_token_mask] = float('-inf')
#     topk_indices_end = torch.topk(attr_end_masked, int(attr_end.shape[0] * topk / 100), sorted=False).indices

#     keep_mask_end = torch.zeros(num_tokens, dtype=torch.bool, device=input_embed.device)
#     keep_mask_end[topk_indices_end] = True
#     keep_mask_end[special_token_mask] = True

#     perturbed_embed_end = input_embed.detach().clone()
#     perturbed_embed_end[0, ~keep_mask_end, :] = base_token_emb

#     with torch.no_grad():
#         outputs_pert_end = model(inputs_embeds=perturbed_embed_end, attention_mask=attention_mask, **extra_kwargs)
#         end_logits_pert = outputs_pert_end.end_logits[0]
#         prob_end_pert = F.softmax(end_logits_pert, dim=0)[end_idx]

#     suff_end = (prob_end_orig - prob_end_pert).item()

#     return suff_start, suff_end

# def eval_wae(scaled_features, word_embedding, epsilon=0.1):
# 	"""
# 	Compute Word Alignment Error (WAE).
	
# 	Measures the smallest distance of each embedding to any word embedding
# 	and reports the average among all words in the path for a sentence.
# 	"""
# 	dists = []
# 	for emb in scaled_features:
# 		all_dist = torch.sqrt(torch.sum((word_embedding - emb.unsqueeze(0)) ** 2, dim=1))
# 		dists.append(torch.min(all_dist).item())

# 	return np.mean(dists)