import tensorboard
import packaging

if not hasattr(tensorboard, "__version__") or packaging.version.parse(
        tensorboard.__version__
) < packaging.version.Version("1.15"):
    raise ImportError("TensorBoard logging requires TensorBoard version 1.15 or above")

del packaging
del tensorboard

from .writer import FileWriter, SummaryWriter  # noqa: F401
from tensorboard.summary.writer.record_writer import RecordWriter  # noqa: F401
