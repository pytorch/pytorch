import torch
from torch.nn.attention._flex_attention import _flex_attention as flex_attention


torch.manual_seed(0)


# Lets create some input tensors
# The input tensor has shape (batch_size, num_heads, seq_len, head_dim)
# Assume a batch size of 2 for inference.
query = torch.randn(2, 8, 4096, 64, device="cuda", dtype=torch.float32)
key = torch.randn(2, 8, 4096, 64, device="cuda", dtype=torch.float32)
value = torch.randn(2, 8, 4096, 64, device="cuda", dtype=torch.float32)


# Lets create a fun new score_modification! I will call this
# Checkerboard. It will reduce the score for neighboring tokens (1 step apart)
# in the sequence. And increase the score for tokens 2 steps apart. For everything
# else, the score will remain the same.

def checkerboard(score, batch, head, token_q, token_kv):
    score = torch.where(torch.abs(token_kv - token_q) == 1, score * 0.5, score)
    score = torch.where(torch.abs(token_kv - token_q) == 2, score * 2.0, score)
    return score


# Lets call flex_attention with this new score modification (eager)
flex_attention_output = flex_attention(query, key, value, score_mod=checkerboard) # Output shape [B, H, N, d] = [1, 8, 4096, 64]


def eagar_flash_decoder(query, key, value, score_mod, Bc=None) -> torch.Tensor:
    """
    Returns O [B, H, N, d]
    """
    # [B, H] parallelized across different CTAs
    # [N, d] is our main consideration
    # Bc is the KV size assigned to each CTA. N/Bc is the number of CTAs running in parallel for a single head.
    if Bc is None:
        Bc = query.shape[2] # fallback to flash attention
    query = query.view(query.shape[0], query.shape[1], 1, query.shape[2], -1) # [B, H, 1, N, d]
    print(query.shape, "Q shape")
    key = key.view(key.shape[0], key.shape[1], key.shape[2]//Bc, Bc, -1) # [B, H, N/Bc, Bc, d]
    print(key.shape, "K shape")
    value = value.view(value.shape[0], value.shape[1], value.shape[2]//Bc, Bc, -1) # [B, H, N/Bc, Bc, d]
    scores = (query @ key.transpose(-2, -1)) # score = QK^T. shape [B, H, N/Bc, N, Bc]. Broadcast Q along N/Bc dim of K

    B = torch.arange(0, scores.size(0), device=scores.device) # B
    H = torch.arange(0, scores.size(1), device=scores.device) # H
    Q = torch.arange(0, scores.size(3), device=scores.device) # N
    KV = torch.arange(0, scores.size(3), device=scores.device).view(scores.size(2), scores.size(-1)) # [N/Bc, Bc]


    #vmap score_mod. start with score_mod: [] -> []
    score_mod = torch.vmap(score_mod, in_dims=(0, None, None, None, 0)) # Bc. (Bc, [], [], [], Bc)
    score_mod = torch.vmap(score_mod, in_dims=(0, None, None, 0, None)) # N ([N, Bc], [], [], N, Bc)
    score_mod = torch.vmap(score_mod, in_dims=(0, None, None, None, 0)) # N/Bc ([N/Bc, N, Bc], [], [], N, [N/Bc, Bc])
    score_mod = torch.vmap(score_mod, in_dims=(0, None, 0, None, None)) # H ([H, N/Bc, N, Bc], [], H, N, [N/Bc, Bc])
    score_mod = torch.vmap(score_mod, in_dims=(0, 0, None, None, None)) # H ([B, H, N/Bc, N, Bc], B, H, N, [N/Bc, Bc])


    scores = score_mod(scores, B, H, Q, KV)



    # Add local exp floating stability.
    score_rowmax = torch.max(scores, dim=-1, keepdim=True).values # [B, H, N/Bc, N, 1] row max for each row per CTA.
    scores_exp = torch.exp(scores - score_rowmax) # [B, H, N/Bc, N, Bc], broadcast rowmax for all elements on the same CTA in the same row. (Bc)
    sumexp = scores_exp.sum(dim=-1) # [B, H, N/Bc, N]
    output = scores_exp @ value # [B, H, N/Bc, N, Bc] x [B, H, N/Bc, Bc, d] -> [B, H, N/Bc, N, d]


    #### Cross CTA reduction
    # Finding True row max.
    rmax = torch.max(score_rowmax, dim=-3, keepdim=True).values # [B, H, 1, N, 1]

    # Logsumexp reduction
    ## Rebase logsumexp from partial max to true rowmax.
    sumexp = sumexp*torch.exp(score_rowmax.squeeze(-1) - rmax.squeeze(-1))
    sumexp_agg = sumexp.sum(dim=-2) #[B, H, N/Bc, N] -> [B, H, N]
    logsumexp = torch.log(sumexp_agg) + rmax.squeeze(2,4)


    # Output reduction
    ## Rebase roxmax from partial max to true rowmax.
    output = output*torch.exp(score_rowmax - rmax)

    # calculate softmax
    output_agg = output.sum(dim=-3) # [B, H, N/Bc, N, d] -> [B, H, N, d]
    output_agg = output_agg.div(sumexp_agg.unsqueeze(-1)) # [B, H, N, d]/[B, H, N, 1]


    print(output_agg[0, 5], "Flex Decoding output")
    print(logsumexp[0, 5], "Flex Decoding logsumexp")

    return output_agg, logsumexp




decoder_output = eagar_flash_decoder(query, key, value, score_mod=checkerboard, Bc=128)[0]
torch.testing.assert_close(flex_attention_output, decoder_output, atol=2e-2, rtol=2e-2)
