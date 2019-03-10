"""A module for visualization with tensorboard
"""

from tensorboard.compat.tensorboard.record_writer import RecordWriter
from .torchvis import TorchVis
from .writer import FileWriter, SummaryWriter
