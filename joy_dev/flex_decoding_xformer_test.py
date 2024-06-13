import torch
from torch._higher_order_ops.flex_attention import flex_attention as flex_attention_hop


@torch.compile
def compiled_flex_attention(query, key, value, score_mod):
    return flex_attention_hop(query, key, value, score_mod)

@torch.compile(backend="aot_eager")
def eager_flex_attention(query, key, value, score_mod):
    return flex_attention_hop(query, key, value, score_mod)


if __name__ == "__main__":
    torch.manual_seed(10)


    # Lets create some input tensors
    # The input tensor has shape (batch_size, num_heads, seq_len, head_dim)
    # Assume a batch size of 2 for inference.
    query = torch.load('../xformers/tensor_q.torch.bfloat16.pt')
    key = torch.load('../xformers/tensor_k.torch.bfloat16.pt')
    value = torch.load('../xformers/tensor_v.torch.bfloat16.pt')


    query = torch.randn(16, 1, 16, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(16, 1, 4096, 128, device="cuda", dtype=torch.bfloat16)
    value = torch.randn(16, 1, 4096, 128, device="cuda", dtype=torch.bfloat16)


    # Lets create a fun new score_modification! I will call this
    # Checkerboard. It will reduce the score for neighboring tokens (1 step apart)
    # in the sequence. And increase the score for tokens 2 steps apart. For everything
    # else, the score will remain the same.

    def score_mod(score, batch, head, token_q, token_kv):
        return score


    # Lets call flex_attention with this new score modification (eager)
    eager_output, eager_logsumexp = eager_flex_attention(query, key, value, score_mod=score_mod) # Output shape [B, H, N, d] = [1, 8, 4096, 64]
    compiled_output, compiled_lse = compiled_flex_attention(query, key, value, score_mod=score_mod)

    xformer_output = torch.load('../xformers/tensor_out.torch.bfloat16.pt')
    print(xformer_output, "Xformer Output")
    print(compiled_output, "Compiled output")
    print(eager_output, "Eager output")

    # torch.testing.assert_close(eager_output, compiled_output, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(xformer_output, compiled_output, atol=2e-4, rtol=2e-2)
