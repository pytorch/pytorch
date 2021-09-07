# Functional DataPipe
from torch.utils.data.datapipes.map.callable import MapperMapDataPipe as Mapper
from torch.utils.data.datapipes.map.combining import ConcaterMapDataPipe as Concater


__all__ = ['Concater', 'Mapper']
