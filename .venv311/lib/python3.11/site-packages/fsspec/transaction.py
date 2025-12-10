from collections import deque


class Transaction:
    """Filesystem transaction write context

    Gathers files for deferred commit or discard, so that several write
    operations can be finalized semi-atomically. This works by having this
    instance as the ``.transaction`` attribute of the given filesystem
    """

    def __init__(self, fs, **kwargs):
        """
        Parameters
        ----------
        fs: FileSystem instance
        """
        self.fs = fs
        self.files = deque()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End transaction and commit, if exit is not due to exception"""
        # only commit if there was no exception
        self.complete(commit=exc_type is None)
        if self.fs:
            self.fs._intrans = False
            self.fs._transaction = None
            self.fs = None

    def start(self):
        """Start a transaction on this FileSystem"""
        self.files = deque()  # clean up after previous failed completions
        self.fs._intrans = True

    def complete(self, commit=True):
        """Finish transaction: commit or discard all deferred files"""
        while self.files:
            f = self.files.popleft()
            if commit:
                f.commit()
            else:
                f.discard()
        self.fs._intrans = False
        self.fs._transaction = None
        self.fs = None


class FileActor:
    def __init__(self):
        self.files = []

    def commit(self):
        for f in self.files:
            f.commit()
        self.files.clear()

    def discard(self):
        for f in self.files:
            f.discard()
        self.files.clear()

    def append(self, f):
        self.files.append(f)


class DaskTransaction(Transaction):
    def __init__(self, fs):
        """
        Parameters
        ----------
        fs: FileSystem instance
        """
        import distributed

        super().__init__(fs)
        client = distributed.default_client()
        self.files = client.submit(FileActor, actor=True).result()

    def complete(self, commit=True):
        """Finish transaction: commit or discard all deferred files"""
        if commit:
            self.files.commit().result()
        else:
            self.files.discard().result()
        self.fs._intrans = False
        self.fs = None
