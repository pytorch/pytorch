# Note: The entire file is in testing phase. Please do not import!
import warnings

from .dataset import Dataset as MapDataset
from .dataset import IterableDataset as IterDataset

from .common import get_file_pathnames_from_root, get_file_binaries_from_pathnames, extract_files_from_pathname_binaries

from .decoder import Decoder
from .decoder import basichandlers as decoder_basichandlers
from .decoder import imagehandler as decoder_imagehandler

from typing import List, Iterable, Union

try:
    import braceexpand
    IMPORTED_BRACEEXPAND = True
except Exception as e:
    IMPORTED_BRACEEXPAND = False

class ListDirFilesMapDataset(MapDataset):
    def __init__(self, root: str = '.', masks: Union[str, List[str]] = '*.tar'):
        super().__init__()
        self.root  : str = root
        self.masks : Union[str, List[str]] = masks


class LoadFilesFromDiskMapDataset(MapDataset):
    def __init__(self, dataset: ListDirFilesMapDataset):
        super().__init__()
        self.dataset = dataset


# Dataset for loading file pathname (path + filename). Yield pathnames from given disk root dir.
# args:
#     root      - root dir
#     mask      - a unix style pattern filter string or string list for file name
#     recursive - whether to recursively iterate the folders
#     abspath   - wheter to return relative pathname or absolute pathname
#     length    - a noiminal length of the dataset
class ListDirFilesIterDataset(IterDataset):
    def __init__(
            self,
            root: str = '.',
            masks: Union[str, List[str]] = '*.tar',
            *,
            recursive: bool = False,
            abspath: bool = False,
            length: int = -1):
        super().__init__()
        self.root : str = root
        self.masks : Union[str, List[str]] = masks
        self.recursive : bool = recursive
        self.abspath : bool = abspath
        self.length : int = length

    def __iter__(self):
        return get_file_pathnames_from_root(self.root, self.masks, self.recursive, self.abspath)

    # return a noiminal length of the dataset
    def __len__(self):
        return self.length


# Dataset for loading file binary from pathnames. Yield tuple(pathname, binary stream) from given pathnames.
# args:
#     input_dataset   - iterable with pathnames
#     masks           - a unix style pattern filter string or string list for file names
#     unzip_tars      - whether to unzip tar file
#     unzip_zips      - whether to unzip zip file
#     unzip_recursive - whether to unzip recursively if unzip is allowed
#     unzip_mask_tars - a unix style patten filter string for tar files which is going to be unzipped
#     unzip_mask_zips - a unix style patten filter string for zip files which is going to be unzipped
#     length          - a nominal length of the dataset
class LoadFilesFromDiskIterDataset(IterDataset):
    def __init__(
            self,
            input_dataset : Iterable,
            masks : Union[str, List[str]] = '',
            *,
            unzip_tars : bool = False,
            unzip_zips : bool = False,
            unzip_recursive : bool = False,
            unzip_mask_tars : Union[str, List[str]] = '',
            unzip_mask_zips : Union[str, List[str]] = '',
            length : int = -1):
        super().__init__()
        self.input_dataset : Iterable = input_dataset
        self.masks : Union[str, List[str]] = masks
        self.unzip_tars : bool = unzip_tars
        self.unzip_zips : bool = unzip_zips
        self.unzip_recursive : bool = unzip_recursive
        self.unzip_mask_tars : Union[str, List[str]] = unzip_mask_tars
        self.unzip_mask_zips : Union[str, List[str]] = unzip_mask_zips
        self.length : int = length

    def __iter__(self):
        return get_file_binaries_from_pathnames(
            self.input_dataset, self.masks, self.unzip_tars, self.unzip_zips, self.unzip_recursive,
            self.unzip_mask_tars, self.unzip_mask_zips)

    def __len__(self):
        return self.length


# Dataset for extracting files from iterable with tuple(pathname, binary stream).
# Yield tuple(pathname, binary stream) with extracted binary stream
# args:
#    input_dataset   - iterable with tuple(pathname, binary stream)
#    masks           - a unix style pattern filter string or string list for file names
#    unzip_tars      - whether to unzip tar file
#    unzip_zips      - whether to unzip zip file
#    unzip_recursive - whether to unzip recursively if unzip is allowed
#    unzip_mask_tars - a unix style patten filter string for tar files which is going to be unzipped
#    unzip_mask_zips - a unix style patten filter string for zip files which is going to be unzipped
#    length          - a nominal length of the dataset
class ExtractFilesIterDataset(IterDataset):
    def __init__(
            self,
            input_dataset : Iterable,
            masks : Union[str, List[str]] = '',
            *,
            unzip_tars : bool = True,
            unzip_zips : bool = True,
            unzip_recursive : bool = True,
            unzip_mask_tars : Union[str, List[str]] = '',
            unzip_mask_zips : Union[str, List[str]] = '',
            length : int = -1):
        super().__init__()
        self.input_dataset : Iterable = input_dataset
        self.masks : Union[str, List[str]] = masks
        self.unzip_tars : bool = unzip_tars
        self.unzip_zips : bool = unzip_zips
        self.unzip_recursive : bool = unzip_recursive
        self.unzip_mask_tars : Union[str, List[str]] = unzip_mask_tars
        self.unzip_mask_zips : Union[str, List[str]] = unzip_mask_zips
        self.length : int = length


    def __iter__(self):
        return extract_files_from_pathname_binaries(
            self.input_dataset, self.masks, self.unzip_tars, self.unzip_zips, self.unzip_recursive,
            self.unzip_mask_tars, self.unzip_mask_zips)

    def __len__(self):
        return self.length


# Dataset for decoding data in different format (eg, json, png, etc)
# Yield tuple(pathname, decoded object)
# args:
#    input_dataset - iterable with tuple(pathname, binary stream or bytes)
#    decoders      - user passed in decoders/handlers, if None, a set of default decoders will be initialized
#    length        - a nominal length of the dataset
class RoutedDecoderIterDataset(IterDataset):
    def __init__(
            self,
            input_dataset : Iterable,
            *,
            decoders : Union[None, List[str]] = None,
            length: int = -1):
        super().__init__()
        self.input_dataset : Iterable = input_dataset
        if decoders:
            self.decoder = Decoder(decoders)
        else:
            self.decoder = Decoder([decoder_basichandlers, decoder_imagehandler('torch')])
        self.length : int = length

    def add_decoder(self, decoder):
        self.decoder.add_decoder(decoder)

    def __iter__(self):
        for data in self.input_dataset:
            pathname = data[0]
            result = self.decoder(data)
            yield (pathname, result[pathname])

    def __len__(self):
        return self.length

# Dataset for expanding input url macro(s) which can be regular url(s) or braced url(s) into regular urls,
# if the url macro is braced url, it will be expanded into regular url(s) only when module `braceexpand`
# is imported, otherwise, the expanding operation will not be performed, braced url itself will be yield back.
# This dataset yield url back in string type
# args:
#    urls_marco - input url strings which can be regular urls or braced urls
#    length     - a nominal length of the dataset
#
# eg. https://www.xyz.com/training-data-{000...045}.tar will be converted to a list of urls
#     from https://www.xyz.com/training-data-000.tar to https://www.xyz.com/training-data-045.tar,
#     yield one url per iteration.
#
# note: This dataset class need python module `braceexpand`
class ExpandUrlMacroIterDataset(IterDataset):
    def __init__(
            self,
            urls_macro : Union[str, List[str]],
            length : int = -1):
        super().__init__()
        self.urls_macro : Union[str, List[str]] = urls_macro if isinstance(urls_macro, list) else [urls_macro]
        self.length : int = length

    def __iter__(self):
        if not IMPORTED_BRACEEXPAND:
            warnings.warn("module `braceexpand` is not imported, url expanding will not happen")

        for url in self.urls_macro:
            if url:
                if IMPORTED_BRACEEXPAND:
                    yield from braceexpand.braceexpand(url)
                else:
                    yield url

    def __len__(self):
        return self.length
