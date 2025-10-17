import torch
import torch.nn as nn


EMBED_DIM = 512


@torch.compile(
    mode="max-autotune-no-cudagraphs",
    dynamic=True,
    fullgraph=True,
)
class Foo(nn.Module):
    def __init__(self):
        super().__init__()
        self.mha = nn.Transformer(
            d_model=EMBED_DIM,
            nhead=2,
            num_decoder_layers=1,
            num_encoder_layers=1,
        )

    def forward(self, src, tgt):
        return self.mha(src, tgt)


mod = Foo().cuda()


def get_inputs(bs, s, t):
    src = torch.randn(s, bs, EMBED_DIM).cuda()
    tgt = torch.randn(t, bs, EMBED_DIM).cuda()
    for t in [src, tgt]:
        torch._dynamo.mark_dynamic(t, 0)
        torch._dynamo.mark_dynamic(t, 1)
    return src, tgt


N_WARMUP = 2
N_ITERS = 10

# hard reset
from torch._functorch._aot_autograd.autograd_cache import AOTAutogradCache
from torch.compiler._cache import CacheArtifactManager


torch._inductor.codecache.FxGraphCache.clear()
AOTAutogradCache.clear()
CacheArtifactManager.clear()
torch._dynamo.reset()
torch._inductor.codecache.PyCodeCache.cache_clear(purge=True)
torch._inductor.config.autotune_remote_cache = False

# inputs
hint_inputs = get_inputs(16, 256, 384)
for _ in range(N_WARMUP):
    for _ in range(5):
        mod(*hint_inputs)
