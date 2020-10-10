from pip._vendor.pkg_resources import yield_lines
from pip._vendor.six import ensure_str

from pip._internal.utils.typing import MYPY_CHECK_RUNNING

if MYPY_CHECK_RUNNING:
    from typing import Dict, Iterable, List


class DictMetadata(object):
    """IMetadataProvider that reads metadata files from a dictionary.
    """
    def __init__(self, metadata):
        # type: (Dict[str, bytes]) -> None
        self._metadata = metadata

    def has_metadata(self, name):
        # type: (str) -> bool
        return name in self._metadata

    def get_metadata(self, name):
        # type: (str) -> str
        try:
            return ensure_str(self._metadata[name])
        except UnicodeDecodeError as e:
            # Mirrors handling done in pkg_resources.NullProvider.
            e.reason += " in {} file".format(name)
            raise

    def get_metadata_lines(self, name):
        # type: (str) -> Iterable[str]
        return yield_lines(self.get_metadata(name))

    def metadata_isdir(self, name):
        # type: (str) -> bool
        return False

    def metadata_listdir(self, name):
        # type: (str) -> List[str]
        return []

    def run_script(self, script_name, namespace):
        # type: (str, str) -> None
        pass
