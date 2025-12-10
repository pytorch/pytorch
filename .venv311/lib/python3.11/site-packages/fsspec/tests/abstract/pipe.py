import pytest


class AbstractPipeTests:
    def test_pipe_exclusive(self, fs, fs_target):
        fs.pipe_file(fs_target, b"data")
        assert fs.cat_file(fs_target) == b"data"
        with pytest.raises(FileExistsError):
            fs.pipe_file(fs_target, b"data", mode="create")
        fs.pipe_file(fs_target, b"new data", mode="overwrite")
        assert fs.cat_file(fs_target) == b"new data"
