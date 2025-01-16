from __future__ import annotations


class ExportTypes:
    """Specifies how the ONNX model is stored."""

    # TODO(justinchuby): Deprecate and remove this class.

    PROTOBUF_FILE = "Saves model in the specified protobuf file."
    ZIP_ARCHIVE = "Saves model in the specified ZIP file (uncompressed)."
    COMPRESSED_ZIP_ARCHIVE = "Saves model in the specified ZIP file (compressed)."
    DIRECTORY = "Saves model in the specified folder."
