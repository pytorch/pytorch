import tensorboard
from packaging import version  # type: ignore[import]

if not hasattr(tensorboard, "__version__") or version.parse(
    tensorboard.__version__
) < version.Version("1.15"):
    raise ImportError("TensorBoard logging requires TensorBoard version 1.15 or above")

del version
del tensorboard

from .writer import FileWriter, SummaryWriter  # noqa: F401
from tensorboard.summary.writer.record_writer import RecordWriter  # noqa: F401
