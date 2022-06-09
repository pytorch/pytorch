import tensorboard
from pkg_resources import packaging  # type: ignore[attr-defined]

if not hasattr(tensorboard, '__version__') or packaging.version.parse(tensorboard.__version__).release < (1, 15):
    raise ImportError("TensorBoard logging requires TensorBoard version 1.15 or above")

del packaging
del tensorboard

from .writer import FileWriter, SummaryWriter  # noqa: F401
from tensorboard.summary.writer.record_writer import RecordWriter  # noqa: F401
