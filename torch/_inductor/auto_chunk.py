from contextlib import contextmanager

import torch.fx.traceback as fx_traceback


@contextmanager
def auto_chunk():
    """Mark operations for deterministic auto-chunking.

    Operations traced within this context will be annotated so that the
    auto-chunker can identify them as chunking amplifier nodes without
    relying on output/input size ratio heuristics. Using this context
    manager automatically enables the auto-chunker pass.

    Example::

        with torch._inductor.auto_chunk.auto_chunk():
            logits = self.linear(x)
        loss = F.cross_entropy(logits, y)
    """
    with fx_traceback.annotate({"auto_chunk": True}):
        yield
