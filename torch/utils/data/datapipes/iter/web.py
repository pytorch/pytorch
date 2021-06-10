from io import IOBase
from typing import Tuple
from urllib.error import HTTPError, URLError
import urllib.request as urllib
from torch.utils.data import IterDataPipe


class WebIterDataPipe(IterDataPipe[Tuple[str, IOBase]]):
    r""" :class:`WebIterDataPipe`

    Iterable DataPipe to load file url(s) (web url(s) pointing to file(s)),
    yield file url and IO stream in a tuple
    args:
        timeout : timeout for web request
    """

    def __init__(self, source_datapipe, timeout=30):
        self.source_datapipe = source_datapipe
        self.timeout = timeout

    def __iter__(self):
        for furl in self.source_datapipe:
            try:
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
