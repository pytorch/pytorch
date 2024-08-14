# mypy: allow-untyped-defs
import gc
import sys
from typing import Any, Dict, List, NamedTuple, Optional, Tuple
import types
import weakref
import json
from tempfile import NamedTemporaryFile
import torch
from torch.cuda._memory_viz import _frames_fmt, _block_extra
import atexit
import logging
logger = logging.getLogger(__name__)

def observe_garbage(observer):
    enabled = True

    def disable():
        # when GC runs during exit, things like `sys` will already be unloaded
        # so we have to disable the callback to avoid hitting errors.
        nonlocal enabled
        enabled = False
    atexit.register(disable)

    def gc_callback(phase, info):
        nonlocal enabled
        if not enabled:
            return
        if phase == "start":
            gc.set_debug(gc.DEBUG_SAVEALL)
        elif phase == "stop":
            orig_trace = sys.getprofile()
            self_return = [False]

            def do_collect(*args, **kwargs):
                nonlocal enabled
                if not self_return[0]:
                    self_return[0] = True
                else:
                    sys.setprofile(orig_trace)
                    enabled = False
                    try:
                        # things in gc.garbage have survived a collection
                        # so to free them we have to collect a generation greater than them
                        # but that might _also_ free other stuff and we don't want to miss
                        # that stuff. So we have to now force gc at the highest level here,
                        # report all of what we found, _then_ we can free it up.
                        if info['generation'] != 2:
                            gc.collect()
                        observer(gc.garbage)
                        gc.garbage.clear()
                        # we have to re-run GC to clean up the cycles
                        # we saved from before.
                        gc.set_debug(0)
                        before = torch.cuda.memory_allocated()
                        gc.collect()
                        after = torch.cuda.memory_allocated()
                        if before != after:
                            logger.warning("CUDA Memory changed during GC, %d bytes freed.", before - after)
                    finally:
                        enabled = True
                if orig_trace is not None:
                    return orig_trace(*args, **kwargs)
            sys.setprofile(do_collect)

    gc.callbacks.append(gc_callback)

    # provide a way to disarm the callback
    def remove():
        gc.callbacks.remove(gc_callback)
    return remove

# Function to visualize cycles adapated from refcycle:
# Copyright 2013 Mark Dickinson
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

def _get_cell_type():
    def f(x=None):
        return lambda: x
    return type(f().__closure__[0])

CellType = _get_cell_type()

def annotated_references(obj):
    """
    Return known information about references held by the given object.

    Returns a mapping from referents to lists of descriptions.  Note that there
    may be more than one edge leading to any particular referent; hence the
    need for a list.  Descriptions are currently strings.

    """
    references: Dict[int, List[str]] = {}

    def add_reference(name, obj):
        references.setdefault(id(obj), []).append(name)

    def add_attrs(*attrs):
        for attr in attrs:
            if hasattr(obj, attr):
                add_reference(attr, getattr(obj, attr))

    def add_cell_references():
        try:
            add_attrs("cell_contents")
        except ValueError:
            # if cell_contents is empty,
            # accessing it raises ValueError
            # in this case there is no object to
            # annotate
            pass

    def add_function_references():
        add_attrs("__defaults__",
                  "__closure__",
                  "__globals__",
                  "__code__",
                  "__name__",
                  "__module__",
                  "__doc__"
                  "__qualname__",
                  "__annotations__",
                  "__kwdefaults__")


    def add_sequence_references():
        for position, item in enumerate(obj):
            add_reference(f"[{position}]", item)

    def add_dict_references():
        for key, value in obj.items():
            add_reference("key", key)
            add_reference(f"[{repr(key)}]", value)

    def add_set_references():
        for elt in obj:
            add_reference("element", elt)

    def add_bound_method_references():
        add_attrs("__self__", "__func__", "im_class")

    def add_weakref_references():
        # For subclasses of weakref, we can't reliably distinguish the
        # callback (if any) from other attributes.
        if type(obj) is weakref.ref:
            referents = gc.get_referents(obj)
            if len(referents) == 1:
                target = referents[0]
                add_reference("__callback__", target)


    def add_frame_references():
        f_locals = obj.f_locals
        add_attrs("f_back", "f_code", "f_builtins", "f_globals", "f_trace", "f_locals")
        # Some badly-behaved code replaces the f_locals dict with
        # something that doesn't support the full dict interface.  So we
        # only continue with the annotation if f_locals is a Python dict.
        if type(f_locals) is dict:
            for name, local in obj.f_locals.items():
                add_reference(f"local {name}", local)

    def add_getset_descriptor_references():
        add_attrs("__objclass__", "__name__", "__doc__")

    type_based_references = {
        tuple: add_sequence_references,
        list: add_sequence_references,
        dict: add_dict_references,
        set: add_set_references,
        frozenset: add_set_references,
        types.FunctionType: add_function_references,
        types.FrameType: add_frame_references,
        CellType: add_cell_references,
        types.MethodType: add_bound_method_references,
        weakref.ref: add_weakref_references,
        types.GetSetDescriptorType: add_getset_descriptor_references,
    }

    for type_ in type(obj).__mro__:
        if type_ in type_based_references:
            type_based_references[type_]()

    add_attrs("__dict__", "__class__")
    if isinstance(obj, type):
        add_attrs("__mro__")

    return references

###############################################################################
# Object annotations.


BASE_TYPES = (int, float, complex, type(None), str, bytes)
FRAME_FILENAME_LIMIT = 32

def object_annotation(obj):
    """
    Return a string to be used for Graphviz nodes.

    The string should be short but as informative as possible.
    """

    def format_sequence(obj):
        body = ','.join(repr(x) if isinstance(x, BASE_TYPES) else type(x).__name__ for i, x in zip(range(8), obj))
        if len(obj) > 8:
            body = f'{body}, ...{len(obj) - 8}'
        return body

    # For basic types, use the repr.
    if isinstance(obj, BASE_TYPES):
        return repr(obj)
    if type(obj).__name__ == 'function':
        return f"function\n{obj.__name__}"
    elif isinstance(obj, types.MethodType):
        try:
            func_name = obj.__func__.__qualname__
        except AttributeError:
            func_name = "<anonymous>"
        return f"instancemethod\n{func_name}"
    elif isinstance(obj, list):
        return f"[{format_sequence(obj)}]"
    elif isinstance(obj, tuple):
        return f"({format_sequence(obj)})"
    elif isinstance(obj, dict):
        return f"dict[{len(obj)}]"
    elif isinstance(obj, types.ModuleType):
        return f"module\n{obj.__name__}"
    elif isinstance(obj, type):
        return f"type\n{obj.__name__}"
    elif isinstance(obj, weakref.ref):
        referent = obj()
        if referent is None:
            return "weakref (dead referent)"
        else:
            return f"weakref to id 0x{id(referent):x}"
    elif isinstance(obj, types.FrameType):
        filename = obj.f_code.co_filename
        if len(filename) > FRAME_FILENAME_LIMIT:
            filename = "..." + filename[-(FRAME_FILENAME_LIMIT - 3):]
        return f"frame\n{filename}:{obj.f_lineno}"
    else:
        return f"object\n{type(obj).__module__}.{type(obj).__name__}"



class Node(NamedTuple):
    label: str
    context: Optional[str]
    root: bool
    referrents: List[Tuple[str, int]]

def create_graph(objects, *, context=None, filter=None):
    if context is None:
        context = cuda_allocation_context()
    if filter is None:
        filter = is_cuda_tensor

    nodes = [Node(object_annotation(obj), context(obj), filter(obj), []) for obj in objects]
    node_referrers: List[List[int]] = [[] for obj in objects]

    id_to_node = {id(obj): i for i, obj in enumerate(objects)}
    for obj in objects:
        fidx = id_to_node[id(obj)]
        f = nodes[fidx]
        references = annotated_references(obj)
        for referrent in gc.get_referents(obj):
            rid = id(referrent)
            tidx = id_to_node.get(rid, None)
            if tidx is None:
                continue
            labels = references.get(rid, ["?"])
            node_referrers[tidx].append(fidx)
            for label in labels:
                f.referrents.append((label, tidx))

    to_search = [i for i, n in enumerate(nodes) if n.root]
    to_keep = set()
    while to_search:
        idx = to_search.pop()
        if idx in to_keep:
            continue
        to_keep.add(idx)
        referrers = node_referrers[idx]
        to_search.extend(referrers)
    id_to_filtered_id: Dict[int, int] = {}
    filtered: List[Any] = []
    for i, n in enumerate(nodes):
        if i in to_keep:
            id_to_filtered_id[i] = len(id_to_filtered_id)
            filtered.append(n)
    for n in filtered:
        n.referrents[:] = [(label, id_to_filtered_id[idx])
                           for (label, idx) in n.referrents
                           if idx in id_to_filtered_id]
    return filtered

def escape(n):
    return json.dumps(n)


def is_cuda_tensor(obj):
    return isinstance(obj, torch.Tensor) and obj.is_cuda and not isinstance(obj, torch._subclasses.FakeTensor)

def cuda_allocation_context():
    snapshot = torch.cuda.memory._snapshot()
    addr_to_frame = {}
    for seg in snapshot['segments']:
        addr = seg['address']
        for blk in seg['blocks']:
            if blk['state'] == 'active_allocated':
                frames, _ = _block_extra(blk)
                addr_to_frame[addr] = frames
            addr += blk['size']

    def object_context(obj):
        if is_cuda_tensor(obj):
            addr = obj.untyped_storage().data_ptr()
            frames = addr_to_frame.get(addr)
            if frames is not None:
                return '\n'.join(_frames_fmt(frames, full_filename=True))
        return None
    return object_context

def to_dot(nodes):
    lines = ["digraph GraphName {", "node [shape=rect];", 'rankdir=LR;']
    for i, n in enumerate(nodes):
        lines.append(f'{i} [label={escape(n.label)}, color={ "red" if n.root else "black"}];')

    for i, f in enumerate(nodes):
        for label, j in f.referrents:
            lines.append(f'{i} -> {j} [label = {escape(label)}]')
    lines.append("}\n")
    return '\n'.join(lines)

_template = """
<!DOCTYPE html>
<html>
<head>
  <style>
    body {
      margin: 0;
      padding: 0;
      overflow: hidden;
    }

    #container {
      display: flex;
      flex-direction: column;
      height: 100vh;
    }

    #main {
      flex: 2;
      overflow: auto;
    }

    #preContainer {
      flex: 1;
      overflow: auto;
    }

    svg {
        overflow: scroll;
    }

    pre {
      margin: 0;
      padding: 10px;
    }
  </style>
</head>
<body>
  <div id="container">
    <div id="main">
    </div>
    <div id="preContainer">
      <pre id="stacktrace">Mouse over tensor objects to see where they were allocated.</pre>
    </div>
  </div>
<script src='https://cdnjs.cloudflare.com/ajax/libs/viz.js/1.8.0/viz-lite.js'></script>
<script>
let dot = $DOT
let image = Viz(dot, {format: 'svg'});
document.getElementById('main').innerHTML = image
$LISTENERS
</script>
</body>
</html>
"""
_listener_template = """
document.getElementById('node{id}').addEventListener('mouseover', function(event) {{
  document.getElementById("stacktrace").textContent = {stack}
}})
"""
def to_html(nodes):
    listeners = []
    for i, n in enumerate(nodes):
        if n.context is None:
            continue
        s = _listener_template.format(id=str(i + 1), stack=escape(f'{n.label}:\n{n.context}'))
        listeners.append(s)
    dot = to_dot(nodes)
    return _template.replace('$DOT', repr(dot)).replace('$LISTENERS', '\n'.join(listeners))

def observe_tensor_cycles(callback):
    torch.cuda.memory._record_memory_history(max_entries=100000)

    def observer(garbage):
        if garbage:
            if not any(is_cuda_tensor(obj) for obj in garbage):
                logger.info("No CUDA Tensors found in garbage")
                return
            callback(to_html(create_graph(garbage)))
    return observe_garbage(observer)


def warn_tensor_cycles():
    """
    Install a warning that reports whenever a cycle that is holding CUDA memory is observed.

    The warning produces an .html file that visualizes the cycle,
    and links it to the stack frame that allocted the CUDA tensor.

    Reference cycles are freed by the cycle collector rather than being cleaned up
    when the objects in the cycle first become unreachable. If a cycle points to a tensor,
    the CUDA memory for that tensor will not be freed until garbage collection runs.
    Accumulation of CUDA allocations can lead to out of memory errors (OOMs), as well as
    non-deterministic allocation behavior which is harder to debug.
    """
    logger.info("Watching Python reference cycles for CUDA Tensors.")

    def write_and_log(html):
        with NamedTemporaryFile('w', suffix='.html', delete=False) as f:
            f.write(html)
            logger.warning('Reference cycle includes a CUDA Tensor see visualization of cycle %s', f.name)
    return observe_tensor_cycles(write_and_log)
