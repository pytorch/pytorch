import torch
from torch.utils._triton import has_triton


if has_triton():
    import triton
    import triton.language as tl

    @triton.jit
    def _get_tid():
        """Get thread ID within a block."""
        return tl.inline_asm_elementwise(
            """
            mov.u32 $0, %tid.x;
            mov.u32 $1, %tid.y;
            mov.u32 $2, %tid.z;
            """,
            "=r,=r,=r",
            [],
            dtype=(tl.uint32, tl.uint32, tl.uint32),
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _get_ntid():
        """Get number of threads in a block."""
        return tl.inline_asm_elementwise(
            """
            mov.u32 $0, %ntid.x;
            mov.u32 $1, %ntid.y;
            mov.u32 $2, %ntid.z;
            """,
            "=r,=r,=r",
            [],
            dtype=(tl.uint32, tl.uint32, tl.uint32),
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _get_flat_tid():
        """Get flattened thread ID within a block."""
        tid_x, tid_y, tid_z = _get_tid()
        ntid_x, ntid_y, _ = _get_ntid()
        return tid_z * ntid_y * ntid_x + tid_y * ntid_x + tid_x

    @triton.jit
    def _get_flat_bid():
        """Get flattened block ID."""
        return (
            tl.program_id(2) * tl.num_programs(1) * tl.num_programs(0)
            + tl.program_id(1) * tl.num_programs(0)
            + tl.program_id(0)
        )

    @triton.jit
    def _send_signal(addrs, sem: tl.constexpr):
        """Send signal to remote device using atomic operations."""
        tl.inline_asm_elementwise(
            f"""
            {{
                .reg .u32   %tmp32_<1>;
                .reg .pred  %p<1>;

                send_signal:
                    atom.global.{sem}.sys.cas.b32 %tmp32_0, [$1], 0, 1;
                    setp.eq.u32 %p0, %tmp32_0, 0;
                    @!%p0 bra send_signal;
            }}
            """,
            "=r, l",
            [addrs],
            dtype=addrs.dtype,
            is_pure=False,
            pack=1,
        )

    @triton.jit
    def _wait_signal(addrs, sem: tl.constexpr):
        """Wait for signal from remote device using atomic operations."""
        tl.inline_asm_elementwise(
            f"""
            {{
                .reg .u32   %tmp32_<1>;
                .reg .pred  %p<1>;

                wait_signal:
                    atom.global.sys.{sem}.cas.b32 %tmp32_0, [$1], 1, 0;
                    setp.eq.u32 %p0, %tmp32_0, 1;
                    @!%p0 bra wait_signal;
            }}
            """,
            "=r, l",
            [addrs],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )

    @triton.jit
    def symm_mem_sync(
        signal_pad_ptrs,
        rank: tl.constexpr,
        world_size: tl.constexpr,
        block_id,
        hasPreviousMemAccess: tl.constexpr = False,
        hasSubsequentMemAccess: tl.constexpr = False,
    ):
        """
        Synchronizes blocks with matching block_id across participating devices.

        Note: the function itself is not a system level barrier/fence. It is a
        building block for expressing different synchronization patterns.

        Pattern 0: Ensures that all writes to symm_mem buffers from previous
        kernels across all devices are visible to the current kernel:

            symm_mem_sync(..., hasPreviousMemAccess=False, hasSubsequentMemAccess=True)

        Pattern 1: Ensures that all writes to symm_mem buffers from the current
        block are visible to all remote blocks with matching blockIdx:

            symm_mem_sync(..., hasPreviousMemAccess=True, hasSubsequentMemAccess=True)

        Pattern 2: Ensures that symm_mem buffers read by the current kernel are safe
        for writing by subsequent kernels across all devices.

            symm_mem_sync(..., hasPreviousMemAccess=True, hasSubsequentMemAccess=False)

        CUDA graph friendliness:

            This barrier operates through atomic operations on a zero-filled signal
            pad, which resets to a zero-filled state after each successful
            synchronization. This design eliminates the need for incrementing a
            flag from host.
        """
        if block_id is None:
            block_id = _get_flat_bid()
        flat_tid = _get_flat_tid()

        remote_ranks = tl.arange(0, world_size)
        signal_pad_ptrs = signal_pad_ptrs.to(tl.pointer_type(tl.uint64))
        remote_signal_pad_addrs = tl.load(signal_pad_ptrs + remote_ranks).to(
            tl.pointer_type(tl.uint32)
        )
        send_addrs = remote_signal_pad_addrs + block_id * world_size + rank

        local_signal_pad_addr = tl.load(signal_pad_ptrs + rank).to(
            tl.pointer_type(tl.uint32)
        )
        wait_addrs = local_signal_pad_addr + block_id * world_size + remote_ranks

        if hasPreviousMemAccess:
            tl.debug_barrier()

        if flat_tid < world_size:
            _send_signal(send_addrs, "release" if hasPreviousMemAccess else "relaxed")
            _wait_signal(wait_addrs, "acquire" if hasSubsequentMemAccess else "relaxed")

        if hasSubsequentMemAccess:
            tl.debug_barrier()

    torch.ops.symm_mem.symm_mem_sync = symm_mem_sync  # type: ignore[attr-defined]

else:

    def symm_mem_sync(*args, **kwargs):  # type: ignore[no-redef]
        raise RuntimeError(
            "symm_mem_sync requires Triton to be installed. "
            "Please install Triton to use this functionality."
        )

    torch.ops.symm_mem.symm_mem_sync = symm_mem_sync  # type: ignore[attr-defined]
