# Owner(s): ["oncall: distributed"]
"""
Hierarchical All-Reduce Triton Kernel using Symmetric Memory Primitives.

This module implements a topology-aware hierarchical all-reduce algorithm
using the symmetric memory abstraction layer. The algorithm consists of
three phases:

Phase 1: Intra-LSA Reduce-Scatter (Pull-based)
   - Each rank reduces its assigned chunk by pulling data from LSA peers.
   - Uses symm_remote_ptr for direct memory access within the LSA domain.

Phase 2: Inter-Domain Ring All-Reduce (Push-based with signaling)
   - Ring reduce-scatter across LSA domains (nodes)
   - Ring all-gather across LSA domains (nodes)
   - Uses symm_put_signal_async for fused data transfer and signaling
   - Uses symm_signal_wait_until for waiting on remote signals

Phase 3: Intra-LSA Broadcast (Push-based)
   - Each rank broadcasts its fully reduced chunk to all LSA peers.
   - Uses symm_remote_ptr for direct memory stores.

The kernel obtains topology information (rank, team_size, lsa_size) directly
from the SymmContext using symm_team_size and symm_team_rank primitives.
"""

from __future__ import annotations

try:
    import triton
    from triton import language as tl

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    triton = None
    tl = None


if TRITON_AVAILABLE:
    from torch._extern_triton._torch_symm_triton import (
        BACKEND_DEFAULT,
        BACKEND_NVSHMEM,
        requires_torch_symm,
        symm_barrier,
        symm_multicast_ptr,
        symm_remote_ptr,
        symm_put_signal_async,
        symm_signal_wait_until,
        symm_team_rank,
        symm_team_size,
        TL_SCOPE_LSA,
        TL_SCOPE_WORLD,
        TL_SIGNAL_CMP_GE,
        TL_SIGNAL_OP_ADD,
    )

    def make_all_reduce_hierarchical_kernel(backend: int):
        """
        Factory function to create a hierarchical all-reduce kernel.

        This creates a Triton kernel decorated with @requires_torch_symm(backend=X)
        that implements a topology-aware hierarchical all-reduce algorithm.

        The algorithm consists of three phases:
        1. Intra-LSA Reduce-Scatter: Pull-based reduction within LSA domain
        2. Inter-Domain Ring All-Reduce: Ring pattern across nodes using signals
        3. Intra-LSA Broadcast: Push to distribute results within LSA domain

        Args:
            backend: Backend hint (BACKEND_DEFAULT or BACKEND_NVSHMEM)

        Returns:
            A decorated Triton kernel function
        """

        @requires_torch_symm(backend=backend)
        @triton.jit
        def all_reduce_hierarchical(
            ctx_ptr,
            ptr_data,
            ptr_scratch,
            num_elements,
            BLOCK_SIZE: tl.constexpr,
            backend_hint: tl.constexpr,
        ):
            """
            Hierarchical all-reduce using symmetric memory primitives.

            This kernel performs a topology-aware all-reduce operation that
            optimizes for multi-node communication patterns using a three-phase
            approach.

            Topology information (rank, team_size, lsa_size) is obtained from the
            SymmContext using symm_team_size and symm_team_rank primitives.

            Args:
                ctx_ptr: Pointer to SymmContext
                ptr_data: Symmetric data buffer (input/output)
                ptr_scratch: Symmetric scratch buffer for inter-domain communication
                num_elements: Total number of float32 elements
                BLOCK_SIZE: Triton block size (constexpr)
                backend_hint: Backend hint (0=DEFAULT, 2=NVSHMEM)
            """
            # -----------------------------------------------------------------
            # 1. Topology & Identity (obtained from context)
            # -----------------------------------------------------------------
            pid = tl.program_id(0)
            num_programs = tl.num_programs(0)

            # Get topology from context using team primitives
            my_rank = symm_team_rank(ctx_ptr, scope=TL_SCOPE_WORLD, backend=backend_hint)
            team_size = symm_team_size(ctx_ptr, scope=TL_SCOPE_WORLD, backend=backend_hint)
            lsa_size = symm_team_size(ctx_ptr, scope=TL_SCOPE_LSA, backend=backend_hint)

            # Derived topology values
            local_rank = symm_team_rank(ctx_ptr, scope=TL_SCOPE_LSA, backend=backend_hint)
            node_id = my_rank // lsa_size
            num_nodes = team_size // lsa_size

            # -----------------------------------------------------------------
            # 2. Data Partitioning & Pointer Setup
            # -----------------------------------------------------------------
            # Each rank owns chunk_size elements
            chunk_size = num_elements // lsa_size
            my_chunk_start = local_rank * chunk_size

            # For Phase 2, we further partition the chunk across nodes
            # Each sub_chunk corresponds to one node's contribution
            sub_chunk_size = chunk_size // num_nodes

            # Cast raw int64 pointer to proper Triton pointer type for tl.load/store
            p_data = ptr_data.to(tl.pointer_type(tl.float32))
            p_scratch = ptr_scratch.to(tl.pointer_type(tl.float32))

            # Initial barrier to ensure all ranks have data ready
            # Use TL_SCOPE_WORLD to synchronize all ranks in the team
            symm_barrier(ctx_ptr, scope=TL_SCOPE_WORLD, backend=backend_hint)

            # -----------------------------------------------------------------
            # Phase 1: Intra-LSA Reduce-Scatter (Pull Model)
            # -----------------------------------------------------------------
            # Each rank reduces its assigned chunk by pulling data from
            # all LSA peers. This uses direct memory access via symm_remote_ptr.
            # After this phase, each rank has the partial sum of its chunk
            # from all GPUs within the same node.
            # -----------------------------------------------------------------
            for off in range(pid * BLOCK_SIZE, chunk_size, num_programs * BLOCK_SIZE):
                abs_off = my_chunk_start + off
                offsets = abs_off + tl.arange(0, BLOCK_SIZE)
                mask = offsets < (my_chunk_start + chunk_size)

                # Load local data as accumulator
                acc = tl.load(p_data + offsets, mask=mask, other=0.0)

                # Pull and accumulate from all LSA peers
                for i in range(1, lsa_size):
                    peer_local_idx = (local_rank + i) % lsa_size
                    peer_global_rank = node_id * lsa_size + peer_local_idx

                    # Get direct pointer to peer's symmetric buffer
                    peer_ptr_raw = symm_remote_ptr(
                        ctx_ptr, ptr_data, peer_global_rank, backend_hint
                    )
                    peer_ptr = peer_ptr_raw.to(tl.pointer_type(tl.float32))
                    val = tl.load(peer_ptr + offsets, mask=mask, other=0.0)
                    acc = acc + val

                # Store reduced result back to local buffer
                tl.store(p_data + offsets, acc, mask=mask)

            # Barrier after reduce-scatter - all ranks must synchronize
            # Use TL_SCOPE_WORLD because even though Phase 1 only pulls from LSA peers,
            # all ranks in the team must complete Phase 1 before Phase 3 broadcast
            symm_barrier(ctx_ptr, scope=TL_SCOPE_WORLD, backend=backend_hint)

            # -----------------------------------------------------------------
            # Phase 2: Inter-Domain Ring All-Reduce
            # -----------------------------------------------------------------
            # Only execute Phase 2 if there are multiple nodes
            # This uses a ring pattern across nodes with signaling
            # -----------------------------------------------------------------
            if num_nodes > 1:
                # Phase 2a: Ring Reduce-Scatter across nodes
                # Each step, send sub_chunk to next node in ring and receive
                # from previous node, accumulating the results
                for step in range(num_nodes - 1):
                    # Determine which sub_chunk to send/receive this step
                    send_chunk_idx = (node_id - step) % num_nodes
                    recv_chunk_idx = (node_id - step - 1) % num_nodes

                    send_offset = my_chunk_start + send_chunk_idx * sub_chunk_size
                    recv_offset = my_chunk_start + recv_chunk_idx * sub_chunk_size

                    # Destination is same local_rank on next node in ring
                    next_node = (node_id + 1) % num_nodes
                    dest_rank = next_node * lsa_size + local_rank

                    # Send our sub_chunk to next node using scratch buffer
                    # Use signal index = local_rank to avoid conflicts
                    # Cast signal value to int64
                    sig_val = tl.cast(step + 1, tl.int64)

                    # Copy data to scratch buffer for sending
                    for off in range(
                        pid * BLOCK_SIZE, sub_chunk_size, num_programs * BLOCK_SIZE
                    ):
                        src_off = send_offset + off
                        offsets = src_off + tl.arange(0, BLOCK_SIZE)
                        mask = offsets < (send_offset + sub_chunk_size)
                        val = tl.load(p_data + offsets, mask=mask, other=0.0)
                        tl.store(
                            p_scratch + off + tl.arange(0, BLOCK_SIZE), val, mask=mask
                        )

                    # Send scratch buffer to dest_rank with signal
                    # Use TL_SIGNAL_OP_ADD for counting
                    symm_put_signal_async(
                        ctx_ptr,
                        ptr_scratch,  # dest buffer (symmetric address)
                        ptr_scratch,  # src buffer (local data in scratch)
                        sub_chunk_size,
                        tl.float32,
                        dest_rank,
                        local_rank,  # signal_index
                        sig_val,  # signal_value (step + 1)
                        TL_SIGNAL_OP_ADD,
                        backend_hint,
                    )

                    # Wait for data from previous node
                    # Compare: signal >= step + 1 (TL_SIGNAL_CMP_GE)
                    symm_signal_wait_until(
                        ctx_ptr,
                        local_rank,  # signal_index
                        TL_SIGNAL_CMP_GE,
                        sig_val,  # wait for step + 1
                        backend_hint,
                    )

                    # Accumulate received data from scratch into our recv chunk
                    for off in range(
                        pid * BLOCK_SIZE, sub_chunk_size, num_programs * BLOCK_SIZE
                    ):
                        recv_off = recv_offset + off
                        offsets = recv_off + tl.arange(0, BLOCK_SIZE)
                        mask = offsets < (recv_offset + sub_chunk_size)
                        local_val = tl.load(p_data + offsets, mask=mask, other=0.0)
                        scratch_offsets = off + tl.arange(0, BLOCK_SIZE)
                        recv_val = tl.load(
                            p_scratch + scratch_offsets, mask=mask, other=0.0
                        )
                        tl.store(p_data + offsets, local_val + recv_val, mask=mask)

                # Barrier after reduce-scatter across nodes
                    # Use TL_SCOPE_WORLD since Phase 2 involves all nodes
                    symm_barrier(ctx_ptr, scope=TL_SCOPE_WORLD, backend=backend_hint)

                # Phase 2b: Ring All-Gather across nodes
                # Each step, send the fully reduced sub_chunk to next node
                for step in range(num_nodes - 1):
                    # Determine which sub_chunk to send this step
                    send_chunk_idx = (node_id - step + 1) % num_nodes
                    recv_chunk_idx = (node_id - step) % num_nodes

                    send_offset = my_chunk_start + send_chunk_idx * sub_chunk_size
                    recv_offset = my_chunk_start + recv_chunk_idx * sub_chunk_size

                    next_node = (node_id + 1) % num_nodes
                    dest_rank = next_node * lsa_size + local_rank

                    # Signal for allgather phase (offset by num_nodes to avoid overlap)
                    sig_val = tl.cast(num_nodes + step + 1, tl.int64)

                    # Copy send data to scratch
                    for off in range(
                        pid * BLOCK_SIZE, sub_chunk_size, num_programs * BLOCK_SIZE
                    ):
                        src_off = send_offset + off
                        offsets = src_off + tl.arange(0, BLOCK_SIZE)
                        mask = offsets < (send_offset + sub_chunk_size)
                        val = tl.load(p_data + offsets, mask=mask, other=0.0)
                        tl.store(
                            p_scratch + off + tl.arange(0, BLOCK_SIZE), val, mask=mask
                        )

                    # Send to next node
                    symm_put_signal_async(
                        ctx_ptr,
                        ptr_scratch,
                        ptr_scratch,
                        sub_chunk_size,
                        tl.float32,
                        dest_rank,
                        local_rank,
                        sig_val,
                        TL_SIGNAL_OP_ADD,
                        backend_hint,
                    )

                    # Wait for data from previous node
                    symm_signal_wait_until(
                        ctx_ptr,
                        local_rank,
                        TL_SIGNAL_CMP_GE,
                        sig_val,
                        backend_hint,
                    )

                    # Copy received data to the recv chunk (no accumulation, just copy)
                    for off in range(
                        pid * BLOCK_SIZE, sub_chunk_size, num_programs * BLOCK_SIZE
                    ):
                        recv_off = recv_offset + off
                        offsets = recv_off + tl.arange(0, BLOCK_SIZE)
                        mask = offsets < (recv_offset + sub_chunk_size)
                        scratch_offsets = off + tl.arange(0, BLOCK_SIZE)
                        val = tl.load(p_scratch + scratch_offsets, mask=mask, other=0.0)
                        tl.store(p_data + offsets, val, mask=mask)

                # Barrier after all-gather across nodes
                    # Use TL_SCOPE_WORLD since Phase 2 involves all nodes
                    symm_barrier(ctx_ptr, scope=TL_SCOPE_WORLD, backend=backend_hint)

            # -----------------------------------------------------------------
            # Phase 3: Intra-LSA Broadcast (Multicast Push with Unicast Fallback)
            # -----------------------------------------------------------------
            # Each rank broadcasts its fully reduced chunk to all LSA peers.
            # Uses hardware multicast if available (via symm_multicast_ptr),
            # otherwise falls back to unicast via symm_remote_ptr.
            # -----------------------------------------------------------------

            # Try to get multicast pointer (returns 0 if not supported)
            # Team is obtained from the context internally
            mc_ptr_raw = symm_multicast_ptr(
                ctx_ptr, ptr_data, backend_hint
            )

            if mc_ptr_raw != 0:
                # Hardware multicast is available - single write broadcasts to all peers
                mc_ptr = mc_ptr_raw.to(tl.pointer_type(tl.float32))

                for off in range(
                    pid * BLOCK_SIZE, chunk_size, num_programs * BLOCK_SIZE
                ):
                    abs_off = my_chunk_start + off
                    offsets = abs_off + tl.arange(0, BLOCK_SIZE)
                    mask = offsets < (my_chunk_start + chunk_size)

                    val = tl.load(p_data + offsets, mask=mask, other=0.0)
                    tl.store(mc_ptr + offsets, val, mask=mask)
            else:
                # Multicast not available - use unicast to each peer
                for i in range(1, lsa_size):
                    peer_local_idx = (local_rank + i) % lsa_size
                    peer_global_rank = node_id * lsa_size + peer_local_idx

                    peer_ptr_raw = symm_remote_ptr(
                        ctx_ptr, ptr_data, peer_global_rank, backend_hint
                    )
                    peer_ptr = peer_ptr_raw.to(tl.pointer_type(tl.float32))

                    for off in range(
                        pid * BLOCK_SIZE, chunk_size, num_programs * BLOCK_SIZE
                    ):
                        abs_off = my_chunk_start + off
                        offsets = abs_off + tl.arange(0, BLOCK_SIZE)
                        mask = offsets < (my_chunk_start + chunk_size)

                        val = tl.load(p_data + offsets, mask=mask, other=0.0)
                        tl.store(peer_ptr + offsets, val, mask=mask)

            # Final barrier to ensure all broadcasts complete
            # Use TL_SCOPE_WORLD to ensure all ranks have received their data
            symm_barrier(ctx_ptr, scope=TL_SCOPE_WORLD, backend=backend_hint)

        return all_reduce_hierarchical

    # Create kernel variants for each backend
    all_reduce_hierarchical_dynamic = make_all_reduce_hierarchical_kernel(
        BACKEND_DEFAULT
    )
    all_reduce_hierarchical_nvshmem = make_all_reduce_hierarchical_kernel(
        BACKEND_NVSHMEM
    )

else:
    # Triton not available - provide stubs

    def make_all_reduce_hierarchical_kernel(backend: int):
        """Stub for when Triton is not available."""
        raise ImportError("Triton is required for make_all_reduce_hierarchical_kernel")

    all_reduce_hierarchical_dynamic = None
    all_reduce_hierarchical_nvshmem = None


__all__ = [
    "make_all_reduce_hierarchical_kernel",
    "all_reduce_hierarchical_dynamic",
    "all_reduce_hierarchical_nvshmem",
]
