import tensorboard
from torch._vendor.packaging.version import Version

if not hasattr(tensorboard, "__version__") or Version(
    tensorboard.__version__
) < Version("1.15"):
    raise ImportError("TensorBoard logging requires TensorBoard version 1.15 or above")

del Version
del tensorboard

from .writer import FileWriter, SummaryWriter
from tensorboard.summary.writer.record_writer import RecordWriter

__all__ = [
    "FileWriter",
    "RecordWriter",
    "SummaryWriter",
]
