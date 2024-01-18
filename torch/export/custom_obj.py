from dataclasses import dataclass
from typing import Optional


__all__ = ["ScriptObjectMeta"]


@dataclass
class ScriptObjectMeta:
    """
    Metadata which is stored on nodes representing ScriptObjects.
    """

    # Key into constants table to retrieve the real ScriptObject.
    constant_name: Optional[str]
