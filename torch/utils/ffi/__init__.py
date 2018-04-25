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


if cffi.__version_info__ < (1, 4, 0):
    raise ImportError("torch.utils.ffi requires cffi version >= 1.4, but "
                      "got " + '.'.join(map(str, cffi.__version_info__)))


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
        if callable(fn):
            locals[symbol] = _wrap_function(fn, _ffi)
        else:
            locals[symbol] = fn
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
        if os.sys.platform == 'win32':
            cuda_include_dirs = glob.glob(os.getenv('CUDA_PATH', '') + '/include')
            cuda_include_dirs += glob.glob(os.getenv('NVTOOLSEXT_PATH', '') + '/include')
        else:
            cuda_include_dirs = glob.glob('/usr/local/cuda/include')
            cuda_include_dirs += glob.glob('/Developer/NVIDIA/CUDA-*/include')
        include_dirs.append(os.path.join(lib_dir, 'include', 'THC'))
        include_dirs.extend(cuda_include_dirs)
    return wrapper_source, include_dirs


def _create_module_dir(base_path, fullname):
    module, _, name = fullname.rpartition('.')
    if not module:
        target_dir = name
    else:
        target_dir = reduce(os.path.join, fullname.split('.'))
    target_dir = os.path.join(base_path, target_dir)
    try:
        os.makedirs(target_dir)
    except os.error:
        pass
    for dirname in _accumulate(fullname.split('.'), os.path.join):
        init_file = os.path.join(base_path, dirname, '__init__.py')
        open(init_file, 'a').close()  # Create file if it doesn't exist yet
    return name, target_dir


def _build_extension(ffi, cffi_wrapper_name, target_dir, verbose):
    try:
        tmpdir = tempfile.mkdtemp()
        ext_suf = '.pyd' if os.sys.platform == 'win32' else '.so'
        libname = cffi_wrapper_name + ext_suf
        outfile = ffi.compile(tmpdir=tmpdir, verbose=verbose, target=libname)
        shutil.copy(outfile, os.path.join(target_dir, libname))
    finally:
        shutil.rmtree(tmpdir)


def _make_python_wrapper(name, cffi_wrapper_name, target_dir):
    py_source = PY_MODULE_TEMPLATE.substitute(name=name,
                                              cffi_wrapper_name=cffi_wrapper_name)
    with open(os.path.join(target_dir, '__init__.py'), 'w') as f:
        f.write(py_source)


def create_extension(name, headers, sources, verbose=True, with_cuda=False,
                     package=False, relative_to='.', **kwargs):
    """Creates and configures a cffi.FFI object, that builds PyTorch extension.

    Arguments:
        name (str): package name. Can be a nested module e.g. ``.ext.my_lib``.
        headers (str or List[str]): list of headers, that contain only exported
            functions
        sources (List[str]): list of sources to compile.
        verbose (bool, optional): if set to ``False``, no output will be printed
            (default: True).
        with_cuda (bool, optional): set to ``True`` to compile with CUDA headers
            (default: False)
        package (bool, optional): set to ``True`` to build in package mode (for modules
            meant to be installed as pip packages) (default: False).
        relative_to (str, optional): path of the build file. Required when
            ``package is True``. It's best to use ``__file__`` for this argument.
        kwargs: additional arguments that are passed to ffi to declare the
            extension. See `Extension API reference`_ for details.

    .. _`Extension API reference`: https://docs.python.org/3/distutils/apiref.html#distutils.core.Extension
    """
    base_path = os.path.abspath(os.path.dirname(relative_to))
    name_suffix, target_dir = _create_module_dir(base_path, name)
    if not package:
        cffi_wrapper_name = '_' + name_suffix
    else:
        cffi_wrapper_name = (name.rpartition('.')[0] +
                             '.{0}._{0}'.format(name_suffix))

    wrapper_source, include_dirs = _setup_wrapper(with_cuda)
    include_dirs.extend(kwargs.pop('include_dirs', []))

    if os.sys.platform == 'win32':
        library_dirs = glob.glob(os.getenv('CUDA_PATH', '') + '/lib/x64')
        library_dirs += glob.glob(os.getenv('NVTOOLSEXT_PATH', '') + '/lib/x64')

        here = os.path.abspath(os.path.dirname(__file__))
        lib_dir = os.path.join(here, '..', '..', 'lib')

        library_dirs.append(os.path.join(lib_dir))
    else:
        library_dirs = []
    library_dirs.extend(kwargs.pop('library_dirs', []))

    if isinstance(headers, str):
        headers = [headers]
    all_headers_source = ''
    for header in headers:
        with open(os.path.join(base_path, header), 'r') as f:
            all_headers_source += f.read() + '\n\n'

    ffi = cffi.FFI()
    sources = [os.path.join(base_path, src) for src in sources]
    ffi.set_source(cffi_wrapper_name, wrapper_source + all_headers_source,
                   sources=sources,
                   include_dirs=include_dirs,
                   library_dirs=library_dirs, **kwargs)
    ffi.cdef(_typedefs + all_headers_source)

    _make_python_wrapper(name_suffix, '_' + name_suffix, target_dir)

    def build():
        _build_extension(ffi, cffi_wrapper_name, target_dir, verbose)
    ffi.build = build
    return ffi


def _wrap_function(function, ffi):
    @wraps(function)
    def safe_call(*args, **kwargs):
        args = tuple(ffi.cast(_torch_to_cffi.get(type(arg), 'void') + '*', arg._cdata)
                     if isinstance(arg, torch.Tensor) or torch.is_storage(arg)
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
