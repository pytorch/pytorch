"""
Wire protocol for the torchmux coordinator.

Extends checkpoint/protocol.py with baton-management ops for multi-GPU
scheduling (OP_RELEASE_BATON, OP_ACQUIRE_BATON).
"""

import importlib.util
import os

_spec = importlib.util.spec_from_file_location(
    "protocol",
    os.path.join(os.path.dirname(__file__), "../../checkpoint/protocol.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

OP_REGISTER = _mod.OP_REGISTER
OP_WAIT_FOR_TURN = _mod.OP_WAIT_FOR_TURN
OP_PREPARE = _mod.OP_PREPARE
OP_RELEASE_GPU = _mod.OP_RELEASE_GPU
OP_DONE = _mod.OP_DONE
ERR_NO_PEERS = _mod.ERR_NO_PEERS
ERR_PEER_GONE = _mod.ERR_PEER_GONE
read_message = _mod.read_message
write_message = _mod.write_message

OP_RELEASE_BATON = "release_baton"
OP_ACQUIRE_BATON = "acquire_baton"
