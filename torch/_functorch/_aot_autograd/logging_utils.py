# mypy: allow-untyped-defs
"""
Contains utils for logging in AOTAutograd, including managing the names of the graphs under
compilation, capturing user-friendly tracebacks, and debug messages.
"""

import collections
from contextlib import contextmanager
from typing import List, Tuple

import torch
import torch.fx.traceback as fx_traceback


# This is a list since looking forward, we can have this arbitrarily nested.
graph_being_compiled: List[str] = []
# TODO: It would be nice to reset the numbering every time aot_id goes
# up, but this is annoying to do right now (because we don't know if
# an aot_id will come back from the dead), so right now this also happens
# to be a globally unique number too (at the cost of wobbling if you change
# how the graphs compile)
nth_graph: int = 0
model_name: str = "model"


def set_model_name(name):
    global model_name
    model_name = name


def get_aot_compilation_context() -> Tuple[List[str], str, int]:
    return list(graph_being_compiled), model_name, nth_graph


def get_aot_graph_name() -> str:
    """
    Returns the name of the graph being compiled.
    """
    global model_name, graph_being_compiled, nth_graph
    return f"{model_name}__{'_'.join(graph_being_compiled)}_{nth_graph}"


get_graph_being_compiled = get_aot_graph_name


@contextmanager
def track_graph_compiling(aot_config, graph_name):
    global graph_being_compiled
    # TODO: Don't shove the aot_id in here; set it in the context
    graph_being_compiled = [f"{aot_config.aot_id}_{graph_name}"]
    old_name = None
    if tracing_context := torch._guards.TracingContext.try_get():
        old_name = tracing_context.aot_graph_name
        tracing_context.aot_graph_name = graph_being_compiled
        has_tracing_context = True
    else:
        has_tracing_context = False
    try:
        yield
    finally:
        global nth_graph
        nth_graph += 1
        graph_being_compiled = []
        if has_tracing_context:
            if tracing_context := torch._guards.TracingContext.try_get():
                tracing_context.aot_graph_name = old_name


# Set up hooks so that during backward the fx's stack_trace is properly set
callback_set = False


def setup_stacktrace_preservation_hooks(roots: List):
    def iter_graph(roots):
        if not roots:
            return
        seen = set()
        q = collections.deque()  # type: ignore[var-annotated]
        for node in roots:
            if node is not None and node not in seen:
                seen.add(node)
                q.append(node)

        while q:
            node = q.popleft()
            for fn, _idx in node.next_functions:
                if fn in seen or fn is None:
                    continue
                seen.add(fn)
                q.append(fn)

            yield node

    def get_callback(saved_stack_):
        def callback():
            global callback_set
            fx_traceback.set_stack_trace(saved_stack_)
            callback_set = False

        return callback

    def get_prehook(stack_, seq_nr):
        def prehook(grad_output):
            global callback_set

            if not callback_set:
                torch.autograd.variable.Variable._execution_engine.queue_callback(  # type: ignore[attr-defined]
                    get_callback(fx_traceback.format_stack())
                )
                callback_set = True

            fx_traceback.set_stack_trace(stack_)
            fx_traceback.set_grad_fn_seq_nr(seq_nr)

        return prehook

    def get_posthook(special_stack_, seq_nr):
        def posthook(grad_input, grad_output):
            fx_traceback.set_stack_trace(special_stack_)
            fx_traceback.reset_grad_fn_seq_nr()

        return posthook

    for node in iter_graph(roots):
        forward_node_stack = node.metadata.get("traceback_", [])
        node.register_prehook(get_prehook(forward_node_stack, node._sequence_nr()))

        special_stack = forward_node_stack.copy()
        special_stack.append(
            "Gradient addition node due to multiple use of tensor around:"
        )
        node.register_hook(get_posthook(special_stack, node._sequence_nr()))


def describe_input(i, aot_config):
    if i < aot_config.num_params_buffers:
        return f"parameter/buffer {i}"
    else:
        return f"input {i - aot_config.num_params_buffers}"


def format_guard_bug_msg(aot_config, expected):
    return (
        f"At compilation time, graph {aot_config.aot_id} was compiled under the "
        f"assumption that {expected}, but at runtime this was not the case.  "
        "This indicates a guard bug in AOTAutograd or Dynamo, please file a bug to PyTorch."
    )
