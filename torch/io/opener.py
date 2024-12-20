class Opener:
    """
    Opener encapsulates file-like object and supports context manager.

    file-like is an object that implements basic io operations, such as
    an opened local file, a read/write buffer, or user-defined object that
    supports specific storage protocol.

    For reader, file-like must implement seek, tell, read and readline.
    For writer, file-like must implement write and flush.
    """
    def __init__(self, file_like):
        self.file_like = file_like

    def __enter__(self):
        return self.file_like

    def __exit__(self, *args):
        pass
