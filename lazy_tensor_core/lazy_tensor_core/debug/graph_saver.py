from __future__ import print_function

import collections
import os
import threading
import lazy_tensor_core

_SAVE_GRAPH_LOCK = threading.Lock()
_SAVE_GRAPH_IDS = collections.defaultdict(dict)


def save_tensors_graph(save_dir, name, tensors):
    fmt = os.environ.get('SAVE_GRAPH_FMT', 'text')
    if fmt == 'text':
        graph = lazy_tensor_core._LAZYC._get_ltc_tensors_text(tensors)
    elif fmt == 'dot':
        graph = lazy_tensor_core._LAZYC._get_ltc_tensors_dot(tensors)
    elif fmt == 'backend':
        graph = lazy_tensor_core._LAZYC._get_ltc_tensors_backend(tensors)
    else:
        raise RuntimeError('Invalid save graph format: {}'.format(fmt))
    tid = threading.current_thread().ident
    with _SAVE_GRAPH_LOCK:
        tdict = _SAVE_GRAPH_IDS[tid]
        id = tdict.get(name, 0)
        tdict[name] = id + 1
    fname = '{}-{}-{}.{}'.format(name, tid, id, fmt)
    with open(os.path.join(save_dir, fname), 'w') as fd:
        fd.write(graph)
