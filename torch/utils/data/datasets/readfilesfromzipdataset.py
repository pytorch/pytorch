from torch.utils.data.dataset import IterableDataset
from torch.utils.data.datasets.common import validate_pathname_binary_tuple
from typing import Iterable, Iterator

import os
import sys
import zipfile
import warnings

class ReadFilesFromZipIterableDataset(IterableDataset):
    r""" :class:`ReadFilesFromZipIterableDataset`.

    IterableDataset to extract zip binary streams from input iterables
    yield pathname and extracted binary stream in a tuple.
    args:
        dataset: Iterable dataset that provides pathname and zip binary stream in tuples
        length: a nominal length of the dataset
    """
    def __init__(
            self,
            dataset : Iterable,
            length : int = -1):
        super().__init__()
        self.dataset : Iterable = dataset
        self.length : int = length

    def __iter__(self) -> Iterator[tuple]:
        if not isinstance(self.dataset, Iterable):
            warnings.warn("dataset must be Iterable type but got {}".format(type(self.dataset)))
            raise TypeError
        for data in self.dataset:
            ret = validate_pathname_binary_tuple(data)
            if ret:
                warnings.warn("got invalid pathname and binary record ({}), abort!".format(ret))
                raise TypeError
            try:
                zips = zipfile.ZipFile(data[1])
                for zipinfo in zips.infolist():
                    # major version should always be 3 here.
                    if sys.version_info[1] >= 6:
                        if zipinfo.is_dir():
                            continue
                    elif zipinfo.filename.endswith('/'):
                        continue

                    extracted_fobj = zips.open(zipinfo)
                    inner_pathname = os.path.normpath(os.path.join(data[0], zipinfo.filename))
                    # Add a reference of the source zipfile into extracted_fobj, so the source
                    # zipfile handle won't be released until all the extracted file objs are destroyed.
                    # Add `# type: ignore` to silence mypy's type checker
                    extracted_fobj.source_zipfile_ref = zips  # type: ignore
                    yield (inner_pathname, extracted_fobj)
            except Exception as e:
                warnings.warn(
                    "Unable to extract files from corrupted zipfile stream {} due to: {}, abort!".format(data[0], e))
                raise e


    def __len__(self):
        if self.length == -1:
            raise NotImplementedError
        return self.length
