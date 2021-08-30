from torch.utils.data.datapipes.iter.callable import (
    CollateIterDataPipe as Collate,
)
from torch.utils.data.datapipes.iter.callable import MapIterDataPipe as Map
from torch.utils.data.datapipes.iter.callable import (
    TransformsIterDataPipe as Transforms,
)
from torch.utils.data.datapipes.iter.combinatorics import (
    SamplerIterDataPipe as Sampler,
)
from torch.utils.data.datapipes.iter.combinatorics import (
    ShuffleIterDataPipe as Shuffle,
)
from torch.utils.data.datapipes.iter.combining import (
    ConcatIterDataPipe as Concat,
)
from torch.utils.data.datapipes.iter.combining import ZipIterDataPipe as Zip
from torch.utils.data.datapipes.iter.dataframes import (
    DFIterDataPipe as DFIterDataPipe,
)
from torch.utils.data.datapipes.iter.grouping import BatchIterDataPipe as Batch
from torch.utils.data.datapipes.iter.grouping import (
    BucketBatchIterDataPipe as BucketBatch,
)
from torch.utils.data.datapipes.iter.grouping import (
    GroupByKeyIterDataPipe as GroupByKey,
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
           'ByKeyGrouper',
           'Collator',
           'Concater',
           'DFIterDataPipe',
           'FileLister',
           'FileLoader',
           'Filter',
           'HttpReader',
           'IterableWrapper',
           'LineReader',
           'Mapper',
           'RoutedDecoder',
           'Sampler',
           'Shuffler',
           'StreamReader',
           'TarArchiveReader',
           'ZipArchiveReader',
           'Zipper']

# Please keep this list sorted
assert __all__ == sorted(__all__)
