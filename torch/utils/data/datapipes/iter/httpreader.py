from io import IOBase
from typing import Sized, Tuple
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
        from requests import HTTPError, RequestException, Session
        for url in self.datapipe:
            try:
                with Session() as session:
                    if self.timeout is None:
                        r = session.get(url, stream=True)
                    else:
                        r = session.get(url, timeout=self.timeout, stream=True)
                return url, r.raw
            except HTTPError as e:
                raise Exception(f"Could not get the file. [HTTP Error] {e.response}.")
            except RequestException as e:
                raise Exception(f"Could not get the file at {url}. [RequestException] {e.response}.")
            except Exception:
                raise

    def __len__(self) -> int:
        if isinstance(self.datapipe, Sized):
            return len(self.datapipe)
        raise TypeError("{} instance doesn't have valid length".format(type(self).__name__))
