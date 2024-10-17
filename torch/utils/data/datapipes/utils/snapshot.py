# mypy: allow-untyped-defs
from torch.utils.data.datapipes._hook_iterator import _SnapshotState
from torch.utils.data.datapipes.datapipe import IterDataPipe
from torch.utils.data.graph_settings import apply_random_seed


# TODO: Caveats
#   1. Caller (either the ReadingService or DataLoader) must pass in the initial RNG
#   2. `in_batch_shuffle` and `bucketbatch` are not compatible with this because they currently
#      lack the option to `set_seed`.
def _simple_graph_snapshot_restoration(
    datapipe: IterDataPipe, n_iterations: int, rng=None
) -> None:
    r"""
    Fast-forward the given DataPipe and its parents by ``n_iterations``, re-doing computations to restore a snapshot.

    For instance, applying this function to the final DataPipe of a graph will restore the snapshot
    (via fast-forward) every DataPipe within the graph.

    After you deserialize a DataPipe, you can use its `_number_of_samples_yielded` attribute as the input
    to this function to forward the DataPipe.

    A DataPipe cannot be restored twice in a row unless there is an iteration started between the restoration
    attempts.

    Note:
        This is the simplest but least efficient way to fast-forward a DataPipe. Usage of other fast-forwarding
        methods (custom ones if necessary) are recommended.

    Args:
        datapipe: IterDataPipe to be fast-forwarded
        n_iterations: number of iterations to fast-forward
        rng: ``Optional[torch.Generator]``. If not ``None``, this RNG will be used for shuffling. The generator
            should be in its `initial` state as it was first passed into ``DataLoader`` or ``ReadingService``.
    """
    if datapipe._snapshot_state == _SnapshotState.Restored:
        raise RuntimeError(
            "Snapshot restoration cannot be applied. You can only restore simple snapshot to the graph "
            "if your graph has not been restored."
        )

    # For this snapshot restoration function, we want the DataPipe to be at its initial state prior to
    # simple fast-forwarding. Therefore, we need to call `reset` twice, because if `SnapshotState` is `Restored`,
    # the first reset will not actually reset.
    datapipe.reset()  # This ensures `SnapshotState` is `Iterating` by this point, even if it was `Restored`.
    apply_random_seed(datapipe, rng)

    remainder = n_iterations
    it = iter(datapipe)  # This always reset the DataPipe if it hasn't already.
    while remainder > 0:
        try:
            next(it)
            remainder -= 1
        except StopIteration as e:
            raise RuntimeError(
                f"Fast-forward {datapipe} by {n_iterations} iterations "
                "exceeds the number of samples available."
            ) from e
    datapipe._fast_forward_iterator = it
    # While the DataPipe has `_fast_forward_iterator`, `next()` will get result from there instead of elsewhere.

    # This will prevent the DataPipe from resetting in the `iter()` call
    # If another DataPipe is consuming it, it won't have to start over again
    datapipe._snapshot_state = _SnapshotState.Restored
