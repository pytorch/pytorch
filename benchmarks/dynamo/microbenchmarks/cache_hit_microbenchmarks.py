import os
import timeit

import torch.fx
from torch._dynamo.utils import counters
from torch._inductor.utils import clear_inductor_caches, fresh_inductor_cache


N = 10000
K = 100


def huge_graph(x):
    for _ in range(N):
        x = x.sin()
    return x


def main():
    torch._inductor.config.fx_graph_cache = True
    torch._inductor.config.fx_graph_remote_cache = False

    with fresh_inductor_cache():
        a = torch.randn(4).cuda()
        compiled_fn = torch.compile(huge_graph, backend="inductor")

        # write to cache
        compiled_fn(a)
        assert counters["inductor"]["fxgraph_cache_miss"] == 1

        def setup():
            torch._dynamo.reset()
            clear_inductor_caches()
            for m in torch._inductor.codecache.PyCodeCache.cache.values():
                os.remove(m.__file__)
            counters.clear()

        def fn():
            result = compiled_fn(a)
            assert counters["inductor"]["fxgraph_cache_miss"] == 0
            assert counters["inductor"]["fxgraph_cache_hit"] == 1
            return result

        t = min(timeit.repeat(fn, setup=setup, number=K, repeat=3))
        print(f"took {t:.1f}s")


if __name__ == "__main__":
    main()
