from distutils.version import LooseVersion

import tensorboard

if not hasattr(tensorboard, '__version__') or LooseVersion(tensorboard.__version__) < LooseVersion('1.15'):
    raise ImportError('TensorBoard logging requires TensorBoard version 1.15 or above')
del LooseVersion
del tensorboard

from tensorboard.summary.writer.record_writer import RecordWriter  # noqa F401

from .writer import FileWriter, SummaryWriter  # noqa F401
