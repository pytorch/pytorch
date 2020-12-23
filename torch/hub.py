import errno
import hashlib
import os
import re
import shutil
import sys
import tempfile
import torch
import warnings
import zipfile

from urllib.request import urlopen, Request
from urllib.parse import urlparse  # noqa: F401

try:
    from tqdm.auto import tqdm  # automatically select proper tqdm submodule if available
except ImportError:
    try:
        from tqdm import tqdm
    except ImportError:
        # fake tqdm if it's not installed
        class tqdm(object):  # type: ignore

            def __init__(self, total=None, disable=False,
                         unit=None, unit_scale=None, unit_divisor=None):
                self.total = total
                self.disable = disable
                self.n = 0
                # ignore unit, unit_scale, unit_divisor; they're just for real tqdm

            def update(self, n):
                if self.disable:
                    return

                self.n += n
                if self.total is None:
                    sys.stderr.write("\r{0:.1f} bytes".format(self.n))
                else:
                    sys.stderr.write("\r{0:.1f}%".format(100 * self.n / float(self.total)))
                sys.stderr.flush()

            def close(self):
                self.disable = True

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.disable:
                    return

                sys.stderr.write('\n')

# matches bfd8deac from resnet18-bfd8deac.pth
HASH_REGEX = re.compile(r'-([a-f0-9]*)\.')

MASTER_BRANCH = 'master'
ENV_TORCH_HOME = 'TORCH_HOME'
ENV_XDG_CACHE_HOME = 'XDG_CACHE_HOME'
DEFAULT_CACHE_DIR = '~/.cache'
VAR_DEPENDENCY = 'dependencies'
MODULE_HUBCONF = 'hubconf.py'
READ_DATA_CHUNK = 8192
_hub_dir = None


# Copied from tools/shared/module_loader to be included in torch package
def import_module(name, path):
    import importlib.util
    from importlib.abc import Loader
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert isinstance(spec.loader, Loader)
    spec.loader.exec_module(module)
    return module


def _remove_if_exists(path):
    if os.path.exists(path):
        if os.path.isfile(path):
            os.remove(path)
        else:
            shutil.rmtree(path)


def _git_archive_link(repo_owner, repo_name, branch):
    return 'https://github.com/{}/{}/archive/{}.zip'.format(repo_owner, repo_name, branch)


def _load_attr_from_module(module, func_name):
    # Check if callable is defined in the module
    if func_name not in dir(module):
        return None
    return getattr(module, func_name)


def _get_torch_home():
    torch_home = os.path.expanduser(
        os.getenv(ENV_TORCH_HOME,
                  os.path.join(os.getenv(ENV_XDG_CACHE_HOME,
                                         DEFAULT_CACHE_DIR), 'torch')))
    return torch_home


def _parse_repo_info(github):
    branch = MASTER_BRANCH
    if ':' in github:
        repo_info, branch = github.split(':')
    else:
        repo_info = github
    repo_owner, repo_name = repo_info.split('/')
    return repo_owner, repo_name, branch


def _get_cache_or_reload(github, force_reload, verbose=True):
    # Setup hub_dir to save downloaded files
    hub_dir = get_dir()
    if not os.path.exists(hub_dir):
        os.makedirs(hub_dir)
    # Parse github repo information
    repo_owner, repo_name, branch = _parse_repo_info(github)
    # Github allows branch name with slash '/',
    # this causes confusion with path on both Linux and Windows.
    # Backslash is not allowed in Github branch name so no need to
    # to worry about it.
    normalized_br = branch.replace('/', '_')
    # Github renames folder repo-v1.x.x to repo-1.x.x
    # We don't know the repo name before downloading the zip file
    # and inspect name from it.
    # To check if cached repo exists, we need to normalize folder names.
    repo_dir = os.path.join(hub_dir, '_'.join([repo_owner, repo_name, normalized_br]))

    use_cache = (not force_reload) and os.path.exists(repo_dir)

    if use_cache:
        if verbose:
            sys.stderr.write('Using cache found in {}\n'.format(repo_dir))
    else:
        cached_file = os.path.join(hub_dir, normalized_br + '.zip')
        _remove_if_exists(cached_file)

        url = _git_archive_link(repo_owner, repo_name, branch)
        sys.stderr.write('Downloading: \"{}\" to {}\n'.format(url, cached_file))
        download_url_to_file(url, cached_file, progress=False)

        with zipfile.ZipFile(cached_file) as cached_zipfile:
            extraced_repo_name = cached_zipfile.infolist()[0].filename
            extracted_repo = os.path.join(hub_dir, extraced_repo_name)
            _remove_if_exists(extracted_repo)
            # Unzip the code and rename the base folder
            cached_zipfile.extractall(hub_dir)

        _remove_if_exists(cached_file)
        _remove_if_exists(repo_dir)
        shutil.move(extracted_repo, repo_dir)  # rename the repo

    return repo_dir


def _check_module_exists(name):
    import importlib.util
    return importlib.util.find_spec(name) is not None


def _check_dependencies(m):
    dependencies = _load_attr_from_module(m, VAR_DEPENDENCY)

    if dependencies is not None:
        missing_deps = [pkg for pkg in dependencies if not _check_module_exists(pkg)]
        if len(missing_deps):
            raise RuntimeError('Missing dependencies: {}'.format(', '.join(missing_deps)))


def _load_entry_from_hubconf(m, model):
    if not isinstance(model, str):
        raise ValueError('Invalid input: model should be a string of function name')

    # Note that if a missing dependency is imported at top level of hubconf, it will
    # throw before this function. It's a chicken and egg situation where we have to
    # load hubconf to know what're the dependencies, but to import hubconf it requires
    # a missing package. This is fine, Python will throw proper error message for users.
    _check_dependencies(m)

    func = _load_attr_from_module(m, model)

    if func is None or not callable(func):
        raise RuntimeError('Cannot find callable {} in hubconf'.format(model))

    return func


def get_dir():
    r"""
    Get the Torch Hub cache directory used for storing downloaded models & weights.

    If :func:`~torch.hub.set_dir` is not called, default path is ``$TORCH_HOME/hub`` where
    environment variable ``$TORCH_HOME`` defaults to ``$XDG_CACHE_HOME/torch``.
    ``$XDG_CACHE_HOME`` follows the X Design Group specification of the Linux
    filesystem layout, with a default value ``~/.cache`` if the environment
    variable is not set.
    """
    # Issue warning to move data if old env is set
    if os.getenv('TORCH_HUB'):
        warnings.warn('TORCH_HUB is deprecated, please use env TORCH_HOME instead')

    if _hub_dir is not None:
        return _hub_dir
    return os.path.join(_get_torch_home(), 'hub')


def set_dir(d):
    r"""
    Optionally set the Torch Hub directory used to save downloaded models & weights.

    Args:
        d (string): path to a local folder to save downloaded models & weights.
    """
    global _hub_dir
    _hub_dir = d


def list(github, force_reload=False):
    r"""
    List all entrypoints available in `github` hubconf.

    Args:
        github (string): a string with format "repo_owner/repo_name[:tag_name]" with an optional
            tag/branch. The default branch is `master` if not specified.
            Example: 'pytorch/vision[:hub]'
        force_reload (bool, optional): whether to discard the existing cache and force a fresh download.
            Default is `False`.
    Returns:
        entrypoints: a list of available entrypoint names

    Example:
        >>> entrypoints = torch.hub.list('pytorch/vision', force_reload=True)
    """
    repo_dir = _get_cache_or_reload(github, force_reload, True)

    sys.path.insert(0, repo_dir)

    hub_module = import_module(MODULE_HUBCONF, repo_dir + '/' + MODULE_HUBCONF)

    sys.path.remove(repo_dir)

    # We take functions starts with '_' as internal helper functions
    entrypoints = [f for f in dir(hub_module) if callable(getattr(hub_module, f)) and not f.startswith('_')]

    return entrypoints


def help(github, model, force_reload=False):
    r"""
    Show the docstring of entrypoint `model`.

    Args:
        github (string): a string with format <repo_owner/repo_name[:tag_name]> with an optional
            tag/branch. The default branch is `master` if not specified.
            Example: 'pytorch/vision[:hub]'
        model (string): a string of entrypoint name defined in repo's hubconf.py
        force_reload (bool, optional): whether to discard the existing cache and force a fresh download.
            Default is `False`.
    Example:
        >>> print(torch.hub.help('pytorch/vision', 'resnet18', force_reload=True))
    """
    repo_dir = _get_cache_or_reload(github, force_reload, True)

    sys.path.insert(0, repo_dir)

    hub_module = import_module(MODULE_HUBCONF, repo_dir + '/' + MODULE_HUBCONF)

    sys.path.remove(repo_dir)

    entry = _load_entry_from_hubconf(hub_module, model)

    return entry.__doc__


# Ideally this should be `def load(github, model, *args, forece_reload=False, **kwargs):`,
# but Python2 complains syntax error for it. We have to skip force_reload in function
# signature here but detect it in kwargs instead.
# TODO: fix it after Python2 EOL
def load(repo_or_dir, model, *args, **kwargs):
    r"""
    Load a model from a github repo or a local directory.

    Note: Loading a model is the typical use case, but this can also be used to
    for loading other objects such as tokenizers, loss functions, etc.

    If :attr:`source` is ``'github'``, :attr:`repo_or_dir` is expected to be
    of the form ``repo_owner/repo_name[:tag_name]`` with an optional
    tag/branch.

    If :attr:`source` is ``'local'``, :attr:`repo_or_dir` is expected to be a
    path to a local directory.

    Args:
        repo_or_dir (string): repo name (``repo_owner/repo_name[:tag_name]``),
            if ``source = 'github'``; or a path to a local directory, if
            ``source = 'local'``.
        model (string): the name of a callable (entrypoint) defined in the
            repo/dir's ``hubconf.py``.
        *args (optional): the corresponding args for callable :attr:`model`.
        source (string, optional): ``'github'`` | ``'local'``. Specifies how
            ``repo_or_dir`` is to be interpreted. Default is ``'github'``.
        force_reload (bool, optional): whether to force a fresh download of
            the github repo unconditionally. Does not have any effect if
            ``source = 'local'``. Default is ``False``.
        verbose (bool, optional): If ``False``, mute messages about hitting
            local caches. Note that the message about first download cannot be
            muted. Does not have any effect if ``source = 'local'``.
            Default is ``True``.
        **kwargs (optional): the corresponding kwargs for callable
            :attr:`model`.

    Returns:
        The output of the :attr:`model` callable when called with the given
        ``*args`` and ``**kwargs``.

    Example:
        >>> # from a github repo
        >>> repo = 'pytorch/vision'
        >>> model = torch.hub.load(repo, 'resnet50', pretrained=True)
        >>> # from a local directory
        >>> path = '/some/local/path/pytorch/vision'
        >>> model = torch.hub.load(path, 'resnet50', pretrained=True)
    """
    source = kwargs.pop('source', 'github').lower()
    force_reload = kwargs.pop('force_reload', False)
    verbose = kwargs.pop('verbose', True)

    if source not in ('github', 'local'):
        raise ValueError(
            f'Unknown source: "{source}". Allowed values: "github" | "local".')

    if source == 'github':
        repo_or_dir = _get_cache_or_reload(repo_or_dir, force_reload, verbose)

    model = _load_local(repo_or_dir, model, *args, **kwargs)
    return model


def _load_local(hubconf_dir, model, *args, **kwargs):
    r"""
    Load a model from a local directory with a ``hubconf.py``.

    Args:
        hubconf_dir (string): path to a local directory that contains a
            ``hubconf.py``.
        model (string): name of an entrypoint defined in the directory's
            `hubconf.py`.
        *args (optional): the corresponding args for callable ``model``.
        **kwargs (optional): the corresponding kwargs for callable ``model``.

    Returns:
        a single model with corresponding pretrained weights.

    Example:
        >>> path = '/some/local/path/pytorch/vision'
        >>> model = _load_local(path, 'resnet50', pretrained=True)
    """
    sys.path.insert(0, hubconf_dir)

    hubconf_path = os.path.join(hubconf_dir, MODULE_HUBCONF)
    hub_module = import_module(MODULE_HUBCONF, hubconf_path)

    entry = _load_entry_from_hubconf(hub_module, model)
    model = entry(*args, **kwargs)

    sys.path.remove(hubconf_dir)

    return model


def download_url_to_file(url, dst, hash_prefix=None, progress=True):
    r"""Download object at the given URL to a local path.

    Args:
        url (string): URL of the object to download
        dst (string): Full path where object will be saved, e.g. `/tmp/temporary_file`
        hash_prefix (string, optional): If not None, the SHA256 downloaded file should start with `hash_prefix`.
            Default: None
        progress (bool, optional): whether or not to display a progress bar to stderr
            Default: True

    Example:
        >>> torch.hub.download_url_to_file('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth', '/tmp/temporary_file')

    """
    file_size = None
    # We use a different API for python2 since urllib(2) doesn't recognize the CA
    # certificates in older Python
    req = Request(url, headers={"User-Agent": "torch.hub"})
    u = urlopen(req)
    meta = u.info()
    if hasattr(meta, 'getheaders'):
        content_length = meta.getheaders("Content-Length")
    else:
        content_length = meta.get_all("Content-Length")
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])

    # We deliberately save it in a temp file and move it after
    # download is complete. This prevents a local working checkpoint
    # being overridden by a broken download.
    dst = os.path.expanduser(dst)
    dst_dir = os.path.dirname(dst)
    f = tempfile.NamedTemporaryFile(delete=False, dir=dst_dir)

    try:
        if hash_prefix is not None:
            sha256 = hashlib.sha256()
        with tqdm(total=file_size, disable=not progress,
                  unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            while True:
                buffer = u.read(8192)
                if len(buffer) == 0:
                    break
                f.write(buffer)
                if hash_prefix is not None:
                    sha256.update(buffer)
                pbar.update(len(buffer))

        f.close()
        if hash_prefix is not None:
            digest = sha256.hexdigest()
            if digest[:len(hash_prefix)] != hash_prefix:
                raise RuntimeError('invalid hash value (expected "{}", got "{}")'
                                   .format(hash_prefix, digest))
        shutil.move(f.name, dst)
    finally:
        f.close()
        if os.path.exists(f.name):
            os.remove(f.name)

def _download_url_to_file(url, dst, hash_prefix=None, progress=True):
    warnings.warn('torch.hub._download_url_to_file has been renamed to\
            torch.hub.download_url_to_file to be a public API,\
            _download_url_to_file will be removed in after 1.3 release')
    download_url_to_file(url, dst, hash_prefix, progress)

# Hub used to support automatically extracts from zipfile manually compressed by users.
# The legacy zip format expects only one file from torch.save() < 1.6 in the zip.
# We should remove this support since zipfile is now default zipfile format for torch.save().
def _is_legacy_zip_format(filename):
    if zipfile.is_zipfile(filename):
        infolist = zipfile.ZipFile(filename).infolist()
        return len(infolist) == 1 and not infolist[0].is_dir()
    return False

def _legacy_zip_load(filename, model_dir, map_location):
    warnings.warn('Falling back to the old format < 1.6. This support will be '
                  'deprecated in favor of default zipfile format introduced in 1.6. '
                  'Please redo torch.save() to save it in the new zipfile format.')
    # Note: extractall() defaults to overwrite file if exists. No need to clean up beforehand.
    #       We deliberately don't handle tarfile here since our legacy serialization format was in tar.
    #       E.g. resnet18-5c106cde.pth which is widely used.
    with zipfile.ZipFile(filename) as f:
        members = f.infolist()
        if len(members) != 1:
            raise RuntimeError('Only one file(not dir) is allowed in the zipfile')
        f.extractall(model_dir)
        extraced_name = members[0].filename
        extracted_file = os.path.join(model_dir, extraced_name)
    return torch.load(extracted_file, map_location=map_location)

def load_state_dict_from_url(url, model_dir=None, map_location=None, progress=True, check_hash=False, file_name=None):
    r"""Loads the Torch serialized object at the given URL.

    If downloaded file is a zip file, it will be automatically
    decompressed.

    If the object is already present in `model_dir`, it's deserialized and
    returned.
    The default value of `model_dir` is ``<hub_dir>/checkpoints`` where
    `hub_dir` is the directory returned by :func:`~torch.hub.get_dir`.

    Args:
        url (string): URL of the object to download
        model_dir (string, optional): directory in which to save the object
        map_location (optional): a function or a dict specifying how to remap storage locations (see torch.load)
        progress (bool, optional): whether or not to display a progress bar to stderr.
            Default: True
        check_hash(bool, optional): If True, the filename part of the URL should follow the naming convention
            ``filename-<sha256>.ext`` where ``<sha256>`` is the first eight or more
            digits of the SHA256 hash of the contents of the file. The hash is used to
            ensure unique names and to verify the contents of the file.
            Default: False
        file_name (string, optional): name for the downloaded file. Filename from `url` will be used if not set.

    Example:
        >>> state_dict = torch.hub.load_state_dict_from_url('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth')

    """
    # Issue warning to move data if old env is set
    if os.getenv('TORCH_MODEL_ZOO'):
        warnings.warn('TORCH_MODEL_ZOO is deprecated, please use env TORCH_HOME instead')

    if model_dir is None:
        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, 'checkpoints')

    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = None
        if check_hash:
            r = HASH_REGEX.search(filename)  # r is Optional[Match[str]]
            hash_prefix = r.group(1) if r else None
        download_url_to_file(url, cached_file, hash_prefix, progress=progress)

    if _is_legacy_zip_format(cached_file):
        return _legacy_zip_load(cached_file, model_dir, map_location)
    return torch.load(cached_file, map_location=map_location)
