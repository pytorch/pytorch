import triton
import triton.language as tl


if hasattr(triton, "aggregate"):

    @triton.jit
    def aggregate_method_helper(x):
        return x + 1

    @triton.aggregate
    class Config:
        size: tl.constexpr

    ConfigAlias = Config

    @triton.aggregate
    class NestedConfig:
        config: "ConfigAlias"

        @triton.jit
        def get_size(self):
            return aggregate_method_helper(self.config.size) - 1

    NestedConfigAlias = NestedConfig

    @triton.jit
    def aggregate_kernel(x, y, N: tl.constexpr):
        cfg = Config(N)
        i = tl.arange(0, 32)
        tl.store(y + i, tl.load(x + i, i < cfg.size, 0) + 1, i < cfg.size)

    @triton.jit
    def nested_aggregate_kernel(x, y, N: tl.constexpr):
        cfg = NestedConfigAlias(ConfigAlias(N))
        i = tl.arange(0, 32)
        size = cfg.get_size()
        tl.store(y + i, tl.load(x + i, i < size, 0) + 1, i < size)
