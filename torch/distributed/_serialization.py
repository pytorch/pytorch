import pickle
from dataclasses import dataclass
from io import BufferedIOBase
from typing import Any

import torch
import torch._weights_only_unpickler as _weights_only_unpickler
from torch.serialization import _load, _save, DEFAULT_PROTOCOL, MAP_LOCATION


__all__: list[str] = []


@dataclass
class _Entry:
    key: str
    is_storage: bool
    length: int


_weights_only_unpickler._add_safe_globals([_Entry])


class _PseudoZipFile:
    def __init__(self) -> None:
        self.records: dict[str, tuple[object, int]] = {}

    def write_record(self, key: str, data: object, length: int) -> None:
        self.records[key] = (data, length)

    def write_to(self, f: BufferedIOBase) -> None:
        entries = []
        for key, (data, length) in self.records.items():
            entries.append(
                _Entry(
                    key=key,
                    is_storage=isinstance(data, torch.UntypedStorage),
                    length=length,
                )
            )

        pickle.dump(entries, f, protocol=DEFAULT_PROTOCOL)

        for data, _ in self.records.values():
            if isinstance(data, bytes):
                f.write(data)
            elif isinstance(data, str):
                f.write(data.encode("utf-8"))
            elif isinstance(data, torch.UntypedStorage):
                data._write_file(f, False, False, 1)
            else:
                raise TypeError(f"unknown type: {type(data)}")

    def read_from(self, f: BufferedIOBase) -> None:
        entries = _weights_only_unpickler.load(f)

        for entry in entries:
            data = f.read(entry.length)
            if entry.is_storage:
                if entry.length == 0:
                    storage = torch.UntypedStorage(0)
                else:
                    storage = torch.frombuffer(
                        data,
                        dtype=torch.uint8,
                    ).untyped_storage()

                self.records[entry.key] = (
                    storage,
                    entry.length,
                )
            else:
                self.records[entry.key] = (data, entry.length)

    def has_record(self, key: str) -> bool:
        return key in self.records

    def get_record(self, key: str) -> object:
        return self.records[key][0]

    def get_storage_from_record(
        self, key: str, _length: int, _type: int
    ) -> torch.Tensor:
        return torch.tensor(self.records[key][0], dtype=torch.uint8)

    def serialization_id(self) -> str:
        return "torchft"


def _streaming_save(
    obj: object,
    f: BufferedIOBase,
    pickle_module: Any = pickle,
    pickle_protocol: int = DEFAULT_PROTOCOL,
) -> None:
    """
    Save the object to a file-like object in a streaming fashion compatible with
    network sockets.

    This behaves similarly to :func:`torch.save` with a few notable differences:

    * A non-seekable file like object can be used when loading.
    * No forwards/backwards compatibility is provided for the serialization
      format. This is only intended to be used with a single version of PyTorch
      with transient storage (i.e. sockets or temp files).
    * mmap is not supported

    See :func:`torch.save` for more details on specific arguments.
    """

    zip_file = _PseudoZipFile()
    _save(
        obj,
        zip_file=zip_file,
        pickle_module=pickle_module,
        pickle_protocol=pickle_protocol,
        _disable_byteorder_record=False,
    )
    zip_file.write_to(f)


def _streaming_load(
    f: BufferedIOBase,
    map_location: MAP_LOCATION = None,
    pickle_module: Any = None,
    *,
    weights_only: bool = True,
    **pickle_load_args: Any,
) -> object:
    """
    Load the object from a file-like object in a streaming fashion compatible with
    network sockets.

    See :func:`_streaming_save` for more details about the streaming behavior.

    See :func:`torch.load` for more details on specific arguments.
    """
    if weights_only:
        if pickle_module is not None:
            raise RuntimeError(
                "Can not safely load weights when explicit pickle_module is specified"
            )
        pickle_module = _weights_only_unpickler
    else:
        if pickle_module is None:
            pickle_module = pickle

    if "encoding" not in pickle_load_args:
        pickle_load_args["encoding"] = "utf-8"

    zip_file = _PseudoZipFile()
    zip_file.read_from(f)
    return _load(
        zip_file=zip_file,
        map_location=map_location,
        pickle_module=pickle_module,
        **pickle_load_args,
    )
