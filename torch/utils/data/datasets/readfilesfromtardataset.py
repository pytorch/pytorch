from torch.utils.data.dataset import IterableDataset
from torch.utils.data.datasets.common import validate_pathname_binary_tuple
from typing import Iterable, Iterator

import os
import tarfile
import warnings

class ReadFilesFromTarIterableDataset(IterableDataset):
    r""" :class:`ReadFilesFromTarIterableDataset`.

    IterableDataset to extract tar binary streams from input iterables
    yield pathname and extracted binary stream in a tuple.
    args:
        dataset: Iterable dataset that provides pathname and tar binary stream in tuples
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
                tar = tarfile.open(fileobj=data[1], mode="r:*")
                for tarinfo in tar:
                    if not tarinfo.isfile():
                        continue
                    extracted_fobj = tar.extractfile(tarinfo)
                    if extracted_fobj is None:
                        warnings.warn("failed to extract file {} from source tarfile {}".format(tarinfo.name, data[0]))
                        raise tarfile.ExtractError
                    inner_pathname = os.path.normpath(os.path.join(data[0], tarinfo.name))
                    # Add a reference of the source tarfile into extracted_fobj, so the source
                    # tarfile handle won't be released until all the extracted file objs are destroyed.
                    # Add `# type: ignore` to silence mypy's type checker
                    extracted_fobj.source_tarfile_ref = tar  # type: ignore
                    yield (inner_pathname, extracted_fobj)
            except Exception as e:
                warnings.warn(
                    "Unable to extract files from corrupted tarfile stream {} due to: {}, abort!".format(data[0], e))
                raise e

    def __len__(self):
        if self.length == -1:
            raise NotImplementedError
        return self.length
