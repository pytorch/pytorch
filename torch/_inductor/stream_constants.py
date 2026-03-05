"""Constants for CUDA stream and event naming in Inductor codegen."""

DEFAULT_STREAM: str = "default_stream"
DEFAULT_STREAM_IDX: int = 0
ENTRANCE_EVENT: str = "event0"
EVENT_NAME_TEMPLATE: str = "event{event_idx:d}"
STREAM_NAME_TEMPLATE: str = "stream{stream_idx:d}"
