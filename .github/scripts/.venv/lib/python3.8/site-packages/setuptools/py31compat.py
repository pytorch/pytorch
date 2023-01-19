__all__ = []

__metaclass__ = type


try:
    # Python >=3.2
    from tempfile import TemporaryDirectory
except ImportError:
    import shutil
    import tempfile

    class TemporaryDirectory:
        """
        Very simple temporary directory context manager.
        Will try to delete afterward, but will also ignore OS and similar
        errors on deletion.
        """

        def __init__(self, **kwargs):
            self.name = None  # Handle mkdtemp raising an exception
            self.name = tempfile.mkdtemp(**kwargs)

        def __enter__(self):
            return self.name

        def __exit__(self, exctype, excvalue, exctrace):
            try:
                shutil.rmtree(self.name, True)
            except OSError:  # removal errors are not the only possible
                pass
            self.name = None
