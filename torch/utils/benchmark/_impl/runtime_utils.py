import contextlib
import typing

import torch


@contextlib.contextmanager
def set_torch_threads(n: int) -> typing.Iterator[None]:
    prior_num_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(n)
        yield
    finally:
        torch.set_num_threads(prior_num_threads)


class ShouldCudaSynchronize:
    """Helper class to determine if CUDA synchronization is needed."""

    @staticmethod
    def cuda_present() -> typing.Optional[bool]:
        if not (torch.has_cuda and torch.cuda.is_available()):
            # CUDA is not present, no need to test.
            return False

        CUDA = torch.profiler.ProfilerActivity.CUDA
        if CUDA not in torch.profiler.supported_activities():
            # Kineto cannot detect GPU events but CUDA is present, so we must
            # always check to be conservative.
            return True

        # We will need to test on a case-by-case basis.
        return None

    def __init__(self):
        self.cuda_detected: bool = False
        self._profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA])

    def __enter__(self):
        self._profiler.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._profiler.__exit__(exc_type, exc_value, traceback)
        self.cuda_detected = bool([
            e for e in self._profiler.events()

            # Profiler does a cuda sync and we're only interested in whether
            # any actual work was scheduled on the GPU.
            if e.name != "cudaDeviceSynchronize"
        ])
