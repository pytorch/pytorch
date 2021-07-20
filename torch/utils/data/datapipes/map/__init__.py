# Functional DataPipe
from torch.utils.data.datapipes.map.callable import MapMapDataPipe as Map
from torch.utils.data.datapipes.map.combining import \
    (ConcatMapDataPipe as Concat)


__all__ = ['Map', 'Concat']
