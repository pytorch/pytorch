import torch._thnn._THNN
from .utils import THNN_H_PATH, parse_header, load_backend
from . import type2backend

generic_functions = parse_header(THNN_H_PATH)
for t in ['Float', 'Double']:
    backend = load_backend(t, torch._thnn._THNN, generic_functions)
    type2backend['torch.' + t + 'Tensor'] = backend
    type2backend[getattr(torch, t + 'Tensor')] = backend

