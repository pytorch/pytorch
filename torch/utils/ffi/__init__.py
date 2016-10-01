import os
import glob
import tempfile
import shutil
from functools import wraps, reduce
from string import Template
import torch
import torch.cuda
from torch._utils import _accumulate

try:
    import cffi
except ImportError:
    raise ImportError("torch.utils.ffi requires the cffi package")


def _generate_typedefs():
    typedefs = []
    for t in ['Double', 'Float', 'Long', 'Int', 'Short', 'Char', 'Byte']:
        for lib in ['TH', 'THCuda']:
            for kind in ['Tensor', 'Storage']:
                python_name = t + kind
                if t == 'Float' and lib == 'THCuda':
                    th_name = 'THCuda' + kind
                else:
                    th_name = lib + t + kind
                th_struct = 'struct ' + th_name

                typedefs += ['typedef {} {};'.format(th_struct, th_name)]
                module = torch if lib == 'TH' else torch.cuda
                python_class = getattr(module, python_name)
                _cffi_to_torch[th_struct] = python_class
                _torch_to_cffi[python_class] = th_struct
    return '\n'.join(typedefs) + '\n'
_cffi_to_torch = {}
_torch_to_cffi = {}
_typedefs = _generate_typedefs()


PY_MODULE_TEMPLATE = Template("""
from torch.utils.ffi import _wrap_function
from .$cffi_wrapper_name import lib as _lib, ffi as _ffi

__all__ = []
def _import_symbols(locals):
    for symbol in dir(_lib):
        fn = getattr(_lib, symbol)
        locals[symbol] = _wrap_function(fn, _ffi)
        __all__.append(symbol)

_import_symbols(locals())
""")


def _setup_wrapper(with_cuda):
    here = os.path.abspath(os.path.dirname(__file__))
    lib_dir = os.path.join(here, '..', '..', 'lib')
    include_dirs = [
        os.path.join(lib_dir, 'include'),
        os.path.join(lib_dir, 'include', 'TH'),
    ]

    wrapper_source = '#include <TH/TH.h>\n'
    if with_cuda:
        import torch.cuda
        wrapper_source += '#include <THC/THC.h>\n'
        cuda_include_dirs = glob.glob('/usr/local/cuda/include')
        cuda_include_dirs += glob.glob('/Developer/NVIDIA/CUDA-*/include')
        include_dirs.append(os.path.join(lib_dir, 'include', 'THC'))
        include_dirs.extend(cuda_include_dirs)
    return wrapper_source, include_dirs


def _create_module_dir(fullname):
    module, _, name = fullname.rpartition('.')
    if not module:
        target_dir = name
    else:
        target_dir = reduce(os.path.join, fullname.split('.'))
    try:
        os.makedirs(target_dir)
    except FileExistsError:
        pass
    for dirname in _accumulate(fullname.split('.'), os.path.join):
        init_file = os.path.join(dirname, '__init__.py')
        open(init_file, 'a').close()  # Create file if it doesn't exist yet
    return name, target_dir


def _build_extension(ffi, cffi_wrapper_name, target_dir, verbose):
    try:
        tmpdir = tempfile.mkdtemp()
        libname = cffi_wrapper_name + '.so'
        ffi.compile(tmpdir=tmpdir, verbose=verbose, target=libname)
        shutil.copy(os.path.join(tmpdir, libname),
                    os.path.join(target_dir, libname))
    finally:
        shutil.rmtree(tmpdir)


def _make_python_wrapper(name, cffi_wrapper_name, target_dir):
    py_source = PY_MODULE_TEMPLATE.substitute(name=name,
            cffi_wrapper_name=cffi_wrapper_name)
    with open(os.path.join(target_dir, '__init__.py'), 'w') as f:
        f.write(py_source)


def compile_extension(name, header, sources=[], verbose=True, with_cuda=False,
        **kwargs):
    name, target_dir = _create_module_dir(name)
    cffi_wrapper_name = '_' + name

    wrapper_source, include_dirs = _setup_wrapper(with_cuda)
    include_dirs.extend(kwargs.pop('include_dirs', []))
    with open(header, 'r') as f:
        header_source = f.read()

    ffi = cffi.FFI()
    sources = [os.path.abspath(src) for src in sources]
    ffi.set_source(cffi_wrapper_name, wrapper_source + header_source,
            sources=sources,
            include_dirs=include_dirs, **kwargs)
    ffi.cdef(_typedefs + header_source);
    _build_extension(ffi, cffi_wrapper_name, target_dir, verbose)
    _make_python_wrapper(name, cffi_wrapper_name, target_dir)


def _wrap_function(function, ffi):
    @wraps(function)
    def safe_call(*args, **kwargs):
        args = tuple(ffi.cast(_torch_to_cffi.get(type(arg), 'void') + '*', arg._cdata)
                if torch.is_tensor(arg) or torch.is_storage(arg)
                else arg
                for arg in args)
        args = (function,) + args
        result = torch._C._safe_call(*args, **kwargs)
        if isinstance(result, ffi.CData):
            typeof = ffi.typeof(result)
            if typeof.kind == 'pointer':
                cdata = int(ffi.cast('uintptr_t', result))
                cname = typeof.item.cname
                if cname in _cffi_to_torch:
                    return _cffi_to_torch[cname](cdata=cdata)
        return result
    return safe_call

