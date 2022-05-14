from torch.utils.data.datapipes.iter.utils import (
    IterableWrapperIterDataPipe as IterableWrapper,
)
from torch.utils.data.datapipes.iter.callable import (
    CollatorIterDataPipe as Collator,
    MapperIterDataPipe as Mapper,
)
from torch.utils.data.datapipes.iter.combinatorics import (
    SamplerIterDataPipe as Sampler,
    ShufflerIterDataPipe as Shuffler,
)
from torch.utils.data.datapipes.iter.combining import (
    ConcaterIterDataPipe as Concater,
    DemultiplexerIterDataPipe as Demultiplexer,
    ForkerIterDataPipe as Forker,
    MultiplexerIterDataPipe as Multiplexer,
    ZipperIterDataPipe as Zipper,
)
from torch.utils.data.datapipes.iter.filelister import (
    FileListerIterDataPipe as FileLister,
)
from torch.utils.data.datapipes.iter.fileopener import (
    FileLoaderIterDataPipe as FileLoader,
    FileOpenerIterDataPipe as FileOpener,
)
from torch.utils.data.datapipes.iter.grouping import (
    BatcherIterDataPipe as Batcher,
    GrouperIterDataPipe as Grouper,
    ShardingFilterIterDataPipe as ShardingFilter,
    UnBatcherIterDataPipe as UnBatcher,
)
from torch.utils.data.datapipes.iter.routeddecoder import (
    RoutedDecoderIterDataPipe as RoutedDecoder,
)
from torch.utils.data.datapipes.iter.selecting import (
    FilterIterDataPipe as Filter,
)
from torch.utils.data.datapipes.iter.streamreader import (
    StreamReaderIterDataPipe as StreamReader,
)

__all__ = ['Batcher',
           'Collator',
           'Concater',
           'Demultiplexer',
           'FileLister',
           'FileLoader',
           'FileOpener',
           'Filter',
           'Forker',
           'Grouper',
           'IterableWrapper',
           'Mapper',
           'Multiplexer',
           'RoutedDecoder',
           'Sampler',
           'ShardingFilter',
           'Shuffler',
           'StreamReader',
           'UnBatcher',
           'Zipper']

# Please keep this list sorted
assert __all__ == sorted(__all__)
