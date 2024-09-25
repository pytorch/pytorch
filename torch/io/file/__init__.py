import io

import torch
from ..opener import Opener

__all__ = [
    "OpenFile",
    "OpenBufferReader",
    "OpenBufferWriter",
    "OpenZipfileReader",
    "OpenZipfileWriterFile",
    "OpenZipfileWriterBufferOrProtocol",
    "check_seekable"
]


class OpenFile(Opener):
    """
    Opener for local disk file
    """
    def __init__(self, name, mode):
        super().__init__(open(name, mode))

    def __exit__(self, *args):
        self.file_like.close()


class OpenBufferReader(Opener):
    """
    Opener for buffer reader
    """
    def __init__(self, buffer):
        super().__init__(buffer)
        check_seekable(buffer)


class OpenBufferWriter(Opener):
    """
    Opener for buffer writer
    """
    def __exit__(self, *args):
        self.file_like.flush()


class OpenZipfileReader(Opener):
    """
    Opener for zipfile reader, file_like can be either path or buffer or
    an opened url-like object
    """
    def __init__(self, file_like) -> None:
        super().__init__(torch._C.PyTorchFileReader(file_like))


class OpenZipfileWriterFile(Opener):
    """
    Opener for local disk zipfile writer
    """
    def __init__(self, name) -> None:
        self.file_stream = None
        self.name = str(name)
        try:
            self.name.encode("ascii")
        except UnicodeEncodeError:
            # PyTorchFileWriter only supports ascii filename.
            # For filenames with non-ascii characters, we rely on Python
            # for writing out the file.
            self.file_stream = io.FileIO(self.name, mode="w")
            super().__init__(torch._C.PyTorchFileWriter(self.file_stream))
        else:
            super().__init__(torch._C.PyTorchFileWriter(self.name))

    def __exit__(self, *args) -> None:
        self.file_like.write_end_of_file()
        if self.file_stream is not None:
            self.file_stream.close()


class OpenZipfileWriterBufferOrProtocol(Opener):
    """
    Opener for buffer or opened url-like object zipfile writer
    """
    def __init__(self, buffer_or_protocol) -> None:
        if not callable(getattr(buffer_or_protocol, "write", None)):
            msg = f"Buffer/Protocol of {str(type(buffer_or_protocol)).strip('<>')} has no callable attribute 'write'"
            if not hasattr(buffer_or_protocol, "write"):
                raise AttributeError(msg)
            raise TypeError(msg)
        self.buffer_or_protocol = buffer_or_protocol
        super().__init__(torch._C.PyTorchFileWriter(buffer_or_protocol))

    def __exit__(self, *args) -> None:
        self.file_like.write_end_of_file()
        self.buffer_or_protocol.flush()


def check_seekable(f) -> bool:
    def raise_err_msg(patterns, e):
        for p in patterns:
            if p in str(e):
                msg = (
                    str(e)
                    + ". You can only torch.load from a file that is seekable."
                    + " Please pre-load the data into a buffer like io.BytesIO and"
                    + " try to load from it instead."
                )
                raise type(e)(msg)
        raise e

    try:
        f.seek(f.tell())
        return True
    except (io.UnsupportedOperation, AttributeError) as e:
        raise_err_msg(["seek", "tell"], e)
    return False
