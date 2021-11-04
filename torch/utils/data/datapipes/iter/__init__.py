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
from torch.utils.data.datapipes.iter.fileloader import (
    FileLoaderIterDataPipe as FileLoader,
)
from torch.utils.data.datapipes.iter.grouping import (
    BatcherIterDataPipe as Batcher,
    BucketBatcherIterDataPipe as BucketBatcher,
    GrouperIterDataPipe as Grouper,
    UnBatcherIterDataPipe as UnBatcher,
)
from torch.utils.data.datapipes.iter.httpreader import (
    HTTPReaderIterDataPipe as HttpReader,
)
from torch.utils.data.datapipes.iter.linereader import (
    LineReaderIterDataPipe as LineReader,
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
from torch.utils.data.datapipes.iter.tararchivereader import (
    TarArchiveReaderIterDataPipe as TarArchiveReader,
)
from torch.utils.data.datapipes.iter.ziparchivereader import (
    ZipArchiveReaderIterDataPipe as ZipArchiveReader,
)
from torch.utils.data.datapipes.iter.utils import (
    IterableWrapperIterDataPipe as IterableWrapper,
)

__all__ = ['Batcher',
           'BucketBatcher',
           'Collator',
           'Concater',
           'DFIterDataPipe',
           'Demultiplexer',
           'FileLister',
           'FileLoader',
           'Filter',
           'Forker',
           'Grouper',
           'HttpReader',
           'IterableWrapper',
           'LineReader',
           'Mapper',
           'Multiplexer',
           'RoutedDecoder',
           'Sampler',
           'Shuffler',
           'StreamReader',
           'TarArchiveReader',
           'UnBatcher',
           'ZipArchiveReader',
           'Zipper']

# Please keep this list sorted
assert __all__ == sorted(__all__)
