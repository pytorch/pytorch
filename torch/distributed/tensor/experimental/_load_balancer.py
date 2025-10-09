# this file contains the `_LoadBalancer` class and its family of implementation
# for different load-balancing strategies in tensor sharding.
from abc import ABC, abstractmethod
from typing import Optional, Union

import torch


# make it private since it's still a prototype
class _LoadBalancer(ABC):
    @abstractmethod
    def _generate_indices(self, restore: bool = False) -> Optional[torch.Tensor]:
        """
        Generate indices for load balancing.
        Args:
            restore (bool):

        Returns:
            The generated indices of shape `(1, seq_len)` if the load-balancing is
            identical within the batch, or `(batch_size, seq_len)` if the load-balancing
            should vary within the batch.

        Warning:
            For Multi-Head Attention, we require the masks over the head dimension are identical
            (i.e. the return value of `_generate_indices()` does not have `heads` dimension).

        Example:
            Here is the causal mask for attention where q_len == kv_len == 8:
                            KV_index
                    [1, 0, 0, 0, 0, 0, 0, 0]
                    [1, 1, 0, 0, 0, 0, 0, 0]
                    [1, 1, 1, 0, 0, 0, 0, 0]
            Q_index [1, 1, 1, 1, 0, 0, 0, 0]
                    [1, 1, 1, 1, 1, 0, 0, 0]
                    [1, 1, 1, 1, 1, 1, 0, 0]
                    [1, 1, 1, 1, 1, 1, 1, 0]
                    [1, 1, 1, 1, 1, 1, 1, 1]

            This mask matrix also represents the computation required to compute
            the masked Q @ K^T by:
            - mask[i, j] == 1: the computation of Q[i, :] dot K[j, :] is required
            - mask[i, j] == 0: the computation should be skipped

            Therefore the number of 1s in matrix represents the amount of computation
            required.

            Assume we want to distribute this Q @ K^T computation to 2 devices, then
            the matrix is also distributed as:
                            KV_index
                    [1, 0, 0, 0, 0, 0, 0, 0]
                    [1, 1, 0, 0, 0, 0, 0, 0]
                    [1, 1, 1, 0, 0, 0, 0, 0]    rank 0
                    [1, 1, 1, 1, 0, 0, 0, 0]
            Q_index ------------------------
                    [1, 1, 1, 1, 1, 0, 0, 0]
                    [1, 1, 1, 1, 1, 1, 0, 0]    rank 1
                    [1, 1, 1, 1, 1, 1, 1, 0]
                    [1, 1, 1, 1, 1, 1, 1, 1]

            An imbalance of computation is observed on these 2 ranks and this could make
            rank 1 the straggler when performing Context Parallel. In order to balance
            the computation, we need to rearrange the QKV tensors before sharding in such a
            way that the result mask matrix is evenly distributed over devices and each
            rank has the number of 1s as close as possible.

            This method defines the strategy of how to rearrange the QKV tensor for better
            load-balance:
            - when `restore == False`, this method returns an indices tensor `rearrange_idx`
            such that Q[rearrange_idx] is the desired Q tensor after rearranging.
            - when `restore == True`, this method returns an indices tensor `restore_idx`
            such that Q[rearrange_idx][restore_idx] == Q, i.e. restoring the rearranged tensor
            back to the original status before rearranging.
        """


class _HeadTailLoadBalancer(_LoadBalancer):
    def __init__(
        self, seq_length: int, world_size: int, device: Union[str, torch.device]
    ):
        self.seq_length = seq_length
        self.world_size = world_size
        self.device = device

    def _generate_indices(self, restore: bool = False) -> torch.Tensor:
        """
        Generate head-and-tail load balancing indices or restore indices.
        Args:
            restore:
                If True, generate restore indices that map head-and-tail rearranged
                positions back to original positions. If False, generate load
                balance indices that rearrange original positions to head-and-tail pattern.

        Returns:
            The generated indices of shape `(1, seq_len)` because the load-balancing is
            identical within the batch.

        Warning:
            For Multi-Head Attention, we require the masks over the head dimension are identical
            (i.e. the return value of `_generate_indices()` does not have `heads` dimension).

        Example:
            Here is the causal mask for attention where q_len == kv_len == 8:
                            KV_index
                    [1, 0, 0, 0, 0, 0, 0, 0]
                    [1, 1, 0, 0, 0, 0, 0, 0]
                    [1, 1, 1, 0, 0, 0, 0, 0]
            Q_index [1, 1, 1, 1, 0, 0, 0, 0]
                    [1, 1, 1, 1, 1, 0, 0, 0]
                    [1, 1, 1, 1, 1, 1, 0, 0]
                    [1, 1, 1, 1, 1, 1, 1, 0]
                    [1, 1, 1, 1, 1, 1, 1, 1]

            Head-tail load-balance strategy rearranges the Q tensor by combining
            Q[0:k] (on seq dim) and Q[-k:] for rank 0, Q[k:2k] and Q[-2k:-k] for
            rank 1, and so on. In python code it looks like:

                k = Q.size(0) // (2 * cp_world_size)
                for rank in range(cp_world_size):
                    reordered_Q[rank * 2 * k : (rank + 1) * 2 * k] = torch.cat(
                        (Q[rank * k : (rank + 1) * k], Q[-(rank + 1) * k : -rank * k])
                    )

            This can also be done by tensor slicing. For the above example, the indices
            tensor for slicing is:
                slice_indices = torch.tensor([0, 7, 1, 6, 2, 5, 3, 4])

            After reordering QKV using the `slice_indices`, the corresponding mask matrix
            distributing over 2 devices becomes well-balanced:
                            KV_index
                    [1, 0, 0, 0, 0, 0, 0, 0]
                    [1, 1, 1, 1, 1, 1, 1, 1]
                    [1, 1, 0, 0, 0, 0, 0, 0]    rank 0
                    [1, 1, 1, 1, 1, 1, 1, 0]
            Q_index ------------------------
                    [1, 1, 1, 0, 0, 0, 0, 0]
                    [1, 1, 1, 1, 1, 1, 0, 0]    rank 1
                    [1, 1, 1, 1, 0, 0, 0, 0]
                    [1, 1, 1, 1, 1, 0, 0, 0]

            To restore the reordering and putting the tensor back, slicing op can do the
            trick with a `restore_indices` such that:
                slice_indices[restore_indices] == torch.tensor([0, 1, 2, ...])

            In this way, `reordered_Q[restore_indices]` will just be the original Q.
        """
        seq_length = self.seq_length
        world_size = self.world_size
        assert seq_length % (world_size * 2) == 0
        chunk_size = seq_length // (world_size * 2)
        all_indices = []

        for rank in range(world_size):
            # Generate indices for first chunk of the cp rank
            first_chunk_start = rank * chunk_size
            first_chunk_indices = list(
                range(first_chunk_start, first_chunk_start + chunk_size)
            )

            # Second chunk: positions from the complementary chunk
            second_chunk_idx = world_size * 2 - rank - 1
            second_chunk_start = second_chunk_idx * chunk_size
            second_chunk_indices = list(
                range(second_chunk_start, second_chunk_start + chunk_size)
            )
            # combine the indices for this rank
            all_indices.extend(first_chunk_indices + second_chunk_indices)

        all_indices_tensor = torch.tensor(
            all_indices, dtype=torch.int, device=self.device
        )
        if restore:
            all_indices_tensor = torch.argsort(all_indices_tensor)

        return all_indices_tensor.unsqueeze(0)  # add batch dim


class _PerDocumentHeadTailLoadBalancer(_LoadBalancer):
    def __init__(
        self,
        seq_length_per_doc: list[list[int]],
        world_size: int,
        device: Union[str, torch.device],
    ):
        """
        `seq_length_per_doc` has size (B, seq_len) if the load-balancing should vary
        within the batch. Otherwise `seq_length_per_doc` should have size (1, seq_len).
        """
        self.seq_length_per_doc = seq_length_per_doc
        self.world_size = world_size
        self.device = device

    def _generate_indices(self, restore: bool = False) -> torch.Tensor:
        """
        Generate the per-document head-and-tail rearrange indices so that after rearranging
        the input is load-balanced in per-document head-and-tail style.

        Args:
            restore:
                If True, generate restore indices that map per-document head-and-tail
                rearranged positions back to original positions. If False, generate load
                balance indices that rearrange original positions to per-document
                head-and-tail pattern.

        Returns:
            The generated indices of shape `(batch_size, seq_len)` if the load-balancing
            should vary within the batch. Otherwise, it should have shape `(1, seq_len)`.

        Warning:
            For Multi-Head Attention, we require the masks over the head dimension are identical
            (i.e. `seq_length_per_doc` must have size (B, seq_len) or (1, seq_len)).

        Example:
            Here is the document causal mask for attention where q_len == kv_len == 16:
                                        KV_index
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            Q_index [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
                    [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
                    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
                    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
                    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0]
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]

            The per-document head-and-tail load-balancer will apply head-and-tail
            reordering within each document. After load-balancing for context-parallel
            on 2 devices, the above mask matrix will look like this:
                                        KV_index
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
                    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
            Q_index [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
                    ------------------------------------------------
                    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
                    [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
                    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0]
        """
        return torch.stack(
            [
                self._generate_indices_for_batch(seq_lengths, restore)
                for seq_lengths in self.seq_length_per_doc
            ]
        )

    def _generate_indices_for_batch(self, seq_length_per_doc, restore) -> torch.Tensor:  # type: ignore[no-untyped-def]
        world_size = self.world_size
        device = self.device
        assert all(
            seq_length % (2 * world_size) == 0 for seq_length in seq_length_per_doc
        )
        chunk_length_per_doc = [
            seq_length // (2 * world_size) for seq_length in seq_length_per_doc
        ]

        indices = []
        document_start_idx = 0
        for seq_length, chunk_length in zip(seq_length_per_doc, chunk_length_per_doc):
            # Generate the indices for the current document
            for rank in range(world_size):
                head_chunk_start_idx = document_start_idx + chunk_length * rank
                tail_chunk_end_idx = document_start_idx + chunk_length * (
                    2 * world_size - rank
                )
                indices.append(
                    torch.arange(
                        head_chunk_start_idx,
                        head_chunk_start_idx + chunk_length,
                        device=device,
                    )
                )
                indices.append(
                    torch.arange(
                        tail_chunk_end_idx - chunk_length,
                        tail_chunk_end_idx,
                        device=device,
                    )
                )

            document_start_idx += seq_length

        indices_tensor = torch.cat(indices)
        if restore:
            indices_tensor = torch.argsort(indices_tensor)

        return indices_tensor


def _create_default_load_balancer(
    seq_length: int, world_size: int, device: Union[str, torch.device]
) -> Optional[_LoadBalancer]:
    from torch.distributed.tensor.experimental._attention import _cp_options

    if _cp_options.enable_load_balance:
        return _HeadTailLoadBalancer(seq_length, world_size, device)
    else:
        return None
