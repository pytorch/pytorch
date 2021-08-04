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
from torch.utils.data.datapipes.iter.listdirfiles import (
    ListDirFilesIterDataPipe as ListDirFiles,
)
from torch.utils.data.datapipes.iter.loadfilesfromdisk import (
    LoadFilesFromDiskIterDataPipe as LoadFilesFromDisk,
)
from torch.utils.data.datapipes.iter.readfilesfromtar import (
    ReadFilesFromTarIterDataPipe as ReadFilesFromTar,
)
from torch.utils.data.datapipes.iter.readfilesfromzip import (
    ReadFilesFromZipIterDataPipe as ReadFilesFromZip,
)
from torch.utils.data.datapipes.iter.readlinesfromfile import (
    ReadLinesFromFileIterDataPipe as ReadLinesFromFile,
)
from torch.utils.data.datapipes.iter.routeddecoder import (
    RoutedDecoderIterDataPipe as RoutedDecoder,
)
from torch.utils.data.datapipes.iter.selecting import (
    FilterIterDataPipe as Filter,
)
from torch.utils.data.datapipes.iter.tobytes import (
    ToBytesIterDataPipe as ToBytes,
)

__all__ = ['Batch',
           'BucketBatch',
           'Collate',
           'Concat',
           'DFIterDataPipe',
           'Filter',
           'GroupByKey',
           'HttpReader',
           'ListDirFiles',
           'LoadFilesFromDisk',
           'Map',
           'ReadFilesFromTar',
           'ReadFilesFromZip',
           'ReadLinesFromFile',
           'RoutedDecoder',
           'Sampler',
           'Shuffle',
           'ToBytes',
           'Transforms',
           'Zip']

# Please keep this list sorted
assert __all__ == sorted(__all__)
