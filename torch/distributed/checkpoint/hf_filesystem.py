import io
import os

try:
    from huggingface_hub import HfFileSystem
except ImportError:
    class HfFileSystem:
        pass


class _HfFileSystem(HfFileSystem):

    def init_path(self, path: str) -> None:
        return path

    def concat_path(
        self, path: str, suffix: str
    ) -> str:
        return os.path.join(path, suffix)


    def create_stream(
        self, path: str, mode: str
    ) -> io.IOBase:
        return self.open(path, mode)
