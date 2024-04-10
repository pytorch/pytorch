import torch

# Not to break other applications, _StreamBase and _EventBase are alias of torch.Stream and torch.Event.
_StreamBase = torch.Stream
_EventBase = torch.Event
