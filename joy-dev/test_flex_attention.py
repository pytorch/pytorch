import torch
from torch.nn.attention._flex_attention import _flex_attention as flex_attention

torch.manual_seed(0)

# Lets create some input tensors
# The input tensor has shape (batch_size, num_heads, seq_len, head_dim)
query = torch.randn(8, 8, 16, 64, device="cuda", dtype=torch.float32)
key = torch.randn(8, 8, 2048, 64, device="cuda", dtype=torch.float32)
value = torch.randn(8, 8, 2048, 64, device="cuda", dtype=torch.float32)

# Lets create a fun new score_modification! I will call this
# Checkerboard. It will reduce the score for neighboring tokens (1 step apart)
# in the sequence. And increase the score for tokens 2 steps apart. For everything
# else, the score will remain the same.

def checkerboard(score, batch, head, token_q, token_kv):
    score = torch.where(torch.abs(token_kv - token_q) == 1, score * 0.5, score)
    score = torch.where(torch.abs(token_kv - token_q) == 2, score * 2.0, score)
    return score

# Lets call flex_attention with this new score modification
output = flex_attention(query, key, value, score_mod=checkerboard) # Output shape [B, H, N, d] = [8, 8, 2048, 64]

print("Run Non compiled flex attention")
compiled_flex_attention = torch.compile(flex_attention)
print("Compile Flex attention")
out_compiled = compiled_flex_attention(query, key, value, score_mod=checkerboard)
print("Run Compiled Flex attention")

## Manual Implementation of Flash Attention
def eager_flash_attention(query, key, value, score_mod) -> torch.Tensor:
    """
    Returns O [B, H, N, d]
    """
    # [B, H] parallelized among different CTAs
    # [N, d] is our main consideration
    # print(key.shape, "K shape") # Size [B(batches), H(heads), N(seq_len), d(head_dim)] = [8, 8, 2048, 64]
    # print(query.shape, "Q shape") # Size [B, H, N, d] = [8, 8, 2048, 64]
    # print(value.shape, "V shape") # Size [B, H, N, d] = [8, 8, 2048, 64]
    scores = (query @ key.transpose(-2, -1)) # score = QK^T. shape [B, H, N, N] = [8, 8, 2048, 2048]


    B = torch.arange(0, scores.size(0), device=scores.device)
    H = torch.arange(0, scores.size(1), device=scores.device)
    Q = torch.arange(0, scores.size(2), device=scores.device)
    KV = torch.arange(0, scores.size(-1), device=scores.device)


    # batched score function, 4dims.
    score_mod = torch.vmap(score_mod, in_dims=(0, None, None, None, 0) )
    score_mod = torch.vmap(score_mod, in_dims=(0, None, None, 0, None) )
    score_mod = torch.vmap(score_mod, in_dims=(0, None, 0, None, None) )
    score_mod = torch.vmap(score_mod, in_dims=(0, 0, None, None, None) )

    scores = score_mod(scores, B, H, Q, KV)

    scores = scores.softmax(dim=-1) # shape [B, H, N, N] = [8, 8, 2048, 2048]


    output = scores @ value # shape [B, H, N, d] = [8, 8, 2048, 64]
    print(output.shape, "O shape")

    return output


manual_out = eager_flash_attention(query, key, value, score_mod=checkerboard)



torch.testing.assert_close(output, out_compiled, atol=2e-2, rtol=2e-2)
