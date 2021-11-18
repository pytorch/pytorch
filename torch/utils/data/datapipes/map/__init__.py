# Functional DataPipe
from torch.utils.data.datapipes.map.callable import MapperMapDataPipe as Mapper
from torch.utils.data.datapipes.map.combinatorics import ShufflerMapDataPipe as Shuffler
from torch.utils.data.datapipes.map.combining import (
    ConcaterMapDataPipe as Concater,
    ZipperMapDataPipe as Zipper
)
from torch.utils.data.datapipes.map.utils import SequenceWrapperMapDataPipe as SequenceWrapper


__all__ = ['Concater', 'Mapper', 'SequenceWrapper', 'Shuffler', 'Zipper']

# Please keep this list sorted
assert __all__ == sorted(__all__)
