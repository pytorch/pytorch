import torch


def foo(x: torch.Tensor) -> None:
    stream = torch.cuda.current_stream()
    x.record_stream(stream)
