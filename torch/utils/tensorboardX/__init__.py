"""A module for visualization with tensorboard
"""

from tensorboard.compat.tensorboardX.record_writer import RecordWriter
from .torchvis import TorchVis
from .writer import FileWriter, SummaryWriter
