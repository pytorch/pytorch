from ..opener import Opener
from .hdfs import HDFSIO


class HDFSOpener(Opener):
    """
    Opener for HDFS storage
    """
    def __init__(self, url, mode):
        file_like = HDFSIO(url, mode)
        super().__init__(file_like)

    def __exit__(self, *args):
        self.file_like.close()
