"""
sys.monitoring integration for TorchDynamo.

This module handles the setup and teardown of sys.monitoring callbacks for
Dynamo's code replacement mechanism. The low-level PY_START callback itself
is implemented in C (eval_frame.c) for performance, but the registration,
event setup, and cleanup logic is handled here in Python for clarity and
maintainability.
"""

import sys
from typing import Optional

# Only available on Python 3.12+. sys is a built-in module, not a package, so
# `import sys.monitoring` fails; access it as an attribute instead.
if sys.version_info >= (3, 12):
    monitoring = sys.monitoring
else:
    monitoring = None  # type: ignore[assignment]


# Tool ID reserved for Dynamo. Using slot 3 (slots 3-5 are available for
# third-party tools; 6 is OPTIMIZER per PEP 669).
DYNAMO_SYS_MONITORING_TOOL_ID = 3

# Track whether we've registered with sys.monitoring yet
_sys_monitoring_registered = False


def is_sys_monitoring_available() -> bool:
    """Check if sys.monitoring is available (Python 3.12+)."""
    return monitoring is not None


def get_py_start_event() -> Optional[int]:
    """Get the PY_START event value from sys.monitoring.

    Returns None if sys.monitoring is not available.
    """
    if monitoring is None:
        return None
    try:
        return monitoring.events.PY_START
    except (AttributeError, ValueError):
        return None


def enable_sys_monitoring(callback_obj) -> bool:
    """Enable sys.monitoring for Dynamo.

    Args:
        callback_obj: The C callback object created by the C code
                     (torch._C._dynamo.eval_frame module)

    Returns:
        True if successfully enabled, False otherwise.
    """
    global _sys_monitoring_registered

    if monitoring is None:
        return False

    try:
        py_start_event = get_py_start_event()
        if py_start_event is None:
            return False

        # Register the tool ID if not already done
        if not _sys_monitoring_registered:
            try:
                monitoring.use_tool_id(DYNAMO_SYS_MONITORING_TOOL_ID, "torch.dynamo")
            except ValueError:
                # Tool ID already in use; that's OK if it's ours
                pass
            _sys_monitoring_registered = True

        # Register the callback for PY_START events
        monitoring.register_callback(
            DYNAMO_SYS_MONITORING_TOOL_ID,
            py_start_event,
            callback_obj,
        )

        # Enable the PY_START event for our tool
        monitoring.set_events(DYNAMO_SYS_MONITORING_TOOL_ID, py_start_event)

        return True
    except Exception:
        return False


def disable_sys_monitoring() -> None:
    """Disable sys.monitoring for Dynamo.

    This clears the callback registration and disables events, but keeps the
    tool ID reservation to avoid churn in sys.monitoring state.
    """
    global _sys_monitoring_registered

    if monitoring is None or not _sys_monitoring_registered:
        return

    try:
        # Disable events first
        monitoring.set_events(DYNAMO_SYS_MONITORING_TOOL_ID, 0)

        # Clear the callback
        py_start_event = get_py_start_event()
        if py_start_event is not None:
            monitoring.register_callback(
                DYNAMO_SYS_MONITORING_TOOL_ID,
                py_start_event,
                None,
            )
    except Exception:
        # Silently ignore errors during cleanup
        pass
