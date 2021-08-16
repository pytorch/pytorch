from torch.utils.data.datapipes.iter.callable import (
    CollatorIterDataPipe as Collator,
    MapperIterDataPipe as Mapper,
    TransformerIterDataPipe as Transformer,
)
from torch.utils.data.datapipes.iter.combinatorics import (
    SamplerIterDataPipe as Sampler,
    ShufflerIterDataPipe as Shuffler,
)
from torch.utils.data.datapipes.iter.combining import (
    ConcaterIterDataPipe as Concater,
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
    ByKeyGrouperIterDataPipe as ByKeyGrouper,
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

__all__ = ['Batcher',
           'BucketBatcher',
           'ByKeyGrouper',
           'Collator',
           'Concater',
           'FileLister',
           'FileLoader',
           'Filter',
           'HttpReader',
           'LineReader',
           'Mapper',
           'RoutedDecoder',
           'Sampler',
           'Shuffler',
           'StreamReader',
           'TarArchiveReader',
           'Transformer',
           'ZipArchiveReader',
           'Zipper']

# Please keep this list sorted
assert __all__ == sorted(__all__)
