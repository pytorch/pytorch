import threading

# Global variable to identify which SubgraphTracer we are in.
# It is sometimes difficult to find an InstructionTranslator to use.
# - This number does not need to be 1:1 with len(tx.output.tracers)
#   (this can happen in a situation where Dynamo gets recursively called)
_current_scope_id = threading.local()


def current_scope_id():
    global _current_scope_id
    if not hasattr(_current_scope_id, "value"):
        _current_scope_id.value = 1
    return _current_scope_id.value


def modify_current_scope_id(delta):
    global _current_scope_id
    _current_scope_id.value = current_scope_id() + delta
