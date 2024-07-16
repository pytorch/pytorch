from ..opener import Opener
from .http import HTTPIO


class HTTPOpener(Opener):
    """
    Opener for HTTP

    HTTP only support load
    """
    def __init__(self, url, mode):
        if 'w' in mode:
            raise ValueError('http only support load')

        file_like = HTTPIO(url)
        super().__init__(file_like)

    def __exit__(self, *args):
        self.file_like.close()
