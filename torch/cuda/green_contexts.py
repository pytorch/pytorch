import torch


__all__ = [
    "GreenContext",
]

_GreenContext = object
SUPPORTED = False

if hasattr(torch._C, "_CUDAGreenContext"):
    _GreenContext = torch._C._CUDAGreenContext  # type: ignore[misc]
    SUPPORTED = True


# Python shim helps Sphinx process docstrings more reliably.
# pyrefly: ignore [invalid-inheritance]
class GreenContext(_GreenContext):
    r"""Wrapper around a CUDA green context.

    .. warning::
       This API is in beta and may change in future releases.
    """

    @staticmethod
    def create(
        *,
        num_sms: int | None = None,
        workqueue_scope: str | None = None,
        workqueue_concurrency_limit: int | None = None,
        device_id: int | None = None,
    ) -> _GreenContext:
        r"""Create a CUDA green context.

        At least one of ``num_sms`` or ``workqueue_scope`` must be specified.
        Both can be combined to partition SMs and configure workqueues in the
        same green context.

        Arguments:
            num_sms (int, optional): The number of SMs to use in the green
                context. When ``None``, SMs are not partitioned.
            workqueue_scope (str, optional): Workqueue sharing scope. One of
                ``"device_ctx"`` (shared across all contexts, default driver
                behaviour) or ``"balanced"`` (non-overlapping workqueues with
                other balanced green contexts). When ``None``, no workqueue
                configuration is applied.
            workqueue_concurrency_limit (int, optional): Maximum number of
                concurrent stream-ordered workloads for the workqueue. Requires
                ``workqueue_scope`` to be set.
            device_id (int, optional): The device index of green context.
                When ``None``, the current device is used.
        """
        if not SUPPORTED:
            raise RuntimeError("PyTorch was not built with Green Context support!")
        return _GreenContext.create(  # type: ignore[attr-defined]
            device_id=device_id,
            num_sms=num_sms,
            workqueue_scope=workqueue_scope,
            workqueue_concurrency_limit=workqueue_concurrency_limit,
        )

    @staticmethod
    def max_workqueue_concurrency(device_id: int | None = None) -> int:
        r"""Return the maximum workqueue concurrency limit for the device.

        This queries the device for the default number of concurrent
        stream-ordered workloads supported by workqueue configuration
        resources.

        Arguments:
            device_id (int, optional): The device index to query. When
                ``None``, the current device is used.
        """
        if not SUPPORTED:
            raise RuntimeError("PyTorch was not built with Green Context support!")
        return _GreenContext.max_workqueue_concurrency(device_id=device_id)  # type: ignore[attr-defined]

    # Note that these functions are bypassed but we define them here
    # for Sphinx documentation purposes
    def set_context(self) -> None:  # pylint: disable=useless-parent-delegation
        r"""Make the green context the current context."""
        return super().set_context()  # type: ignore[misc]

    def pop_context(self) -> None:  # pylint: disable=useless-parent-delegation
        r"""Assuming the green context is the current context, pop it from the
        context stack and restore the previous context.
        """
        return super().pop_context()  # type: ignore[misc]

    def Stream(self) -> "torch.cuda.Stream":
        r"""Return the CUDA Stream used by the green context."""
        return super().Stream()
