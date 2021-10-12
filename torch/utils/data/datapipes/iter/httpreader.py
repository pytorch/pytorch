from io import IOBase
from typing import Sized, Tuple
from urllib.error import HTTPError, URLError
import urllib.request as urllib
from torch.utils.data import IterDataPipe
from torch.utils.data.datapipes.utils.common import deprecation_warning_torchdata


class HTTPReaderIterDataPipe(IterDataPipe[Tuple[str, IOBase]]):
    r""" :class:`HTTPReaderIterDataPipe`

    Iterable DataPipe to load file url(s) (http url(s) pointing to file(s)),
    yield file url and IO stream in a tuple

    Args:
        datapipe: Iterable DataPipe providing urls
        timeout: Timeout for http request
    """

    def __init__(self, datapipe, timeout=None):
        self.datapipe = datapipe
        self.timeout = timeout
        deprecation_warning_torchdata(type(self).__name__)

    def __iter__(self):
        for furl in self.datapipe:
            try:
                if self.timeout is None:
                    r = urllib.urlopen(furl)
                else:
                    r = urllib.urlopen(furl, timeout=self.timeout)

                yield(furl, r)
            except HTTPError as e:
                raise Exception("Could not get the file.\
                                [HTTP Error] {code}: {reason}."
                                .format(code=e.code, reason=e.reason))
            except URLError as e:
                raise Exception("Could not get the file at {url}.\
                                 [URL Error] {reason}."
                                .format(reason=e.reason, url=furl))
            except Exception:
                raise

    def __len__(self) -> int:
        if isinstance(self.datapipe, Sized):
            return len(self.datapipe)
        raise TypeError("{} instance doesn't have valid length".format(type(self).__name__))
