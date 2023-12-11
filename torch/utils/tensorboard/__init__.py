import tensorboard


if not hasattr(tensorboard, "__version__") or (tensorboard.__version__ < "1.15"):
    raise ImportError("TensorBoard logging requires TensorBoard version 1.15 or above")

del tensorboard

from .writer import FileWriter, SummaryWriter  # noqa: F401
from tensorboard.summary.writer.record_writer import RecordWriter  # noqa: F401
