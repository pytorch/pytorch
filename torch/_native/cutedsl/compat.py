try:
    from cutlass.block import block_copy as block_copy
    from cutlass.memory import SmemAllocator as SmemAllocator
except ImportError:
    from cutlass.utils import block_copy as block_copy, SmemAllocator as SmemAllocator
