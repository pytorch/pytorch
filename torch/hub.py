import importlib
import os
import shutil
import sys
import tempfile
import zipfile

if sys.version_info[0] == 2:
    from urlparse import urlparse
    from urllib2 import urlopen  # noqa f811
else:
    from urllib.request import urlopen
    from urllib.parse import urlparse

import torch
import torch.utils.model_zoo as model_zoo

MASTER_BRANCH = 'master'
ENV_TORCH_HUB_DIR = 'TORCH_HUB_DIR'
DEFAULT_TORCH_HUB_DIR = '~/.torch/hub'
READ_DATA_CHUNK = 8192
hub_dir = None


def _check_module_exists(name):
    if sys.version_info >= (3, 4):
        import importlib.util
        return importlib.util.find_spec(name) is not None
    elif sys.version_info >= (3, 3):
        # Special case for python3.3
        import importlib.find_loader
        return importlib.find_loader(name) is not None
    else:
        # NB: imp doesn't handle hierarchical module names (names contains dots).
        try:
            import imp
            imp.find_module(name)
        except Exception:
            return False
        return True


def _remove_if_exists(path):
    if os.path.exists(path):
        if os.path.isfile(path):
            os.remove(path)
        else:
            shutil.rmtree(path)


def _git_archive_link(repo, branch):
    return 'https://github.com/' + repo + '/archive/' + branch + '.zip'


def _download_url_to_file(url, filename):
    sys.stderr.write('Downloading: \"{}\" to {}'.format(url, filename))
    response = urlopen(url)
    with open(filename, 'wb') as f:
        while True:
            data = response.read(READ_DATA_CHUNK)
            if len(data) == 0:
                break
            f.write(data)


def _load_attr_from_module(module_name, func_name):
    m = importlib.import_module(module_name)
    # Check if callable is defined in the module
    if func_name not in dir(m):
        return None
    return getattr(m, func_name)


def set_dir(d):
    r"""
    Optionally set hub_dir to a local dir to save the intermediate model & checkpoint files.
        If this argument is not set, env variable `TORCH_HUB_DIR` will be searched first,
        `~/.torch/hub` will be created and used as fallback.
    """
    global hub_dir
    hub_dir = d


def load(github, model, force_reload=False, *args, **kwargs):
    r"""
    Load a model from a github repo, with pretrained weights.

    Args:
        github: Required, a string with format "repo_owner/repo_name[:tag_name]" with an optional
            tag/branch. The default branch is `master` if not specified.
            Example: 'pytorch/vision[:hub]'
        model: Required, a string of callable name defined in repo's hubconf.py
        force_reload: Optional, whether to discard the existing cache and force a fresh download.
            Default is `False`.
        *args: Optional, the corresponding args for callable `model`.
        **kwargs: Optional, the corresponding kwargs for callable `model`.

    Returns:
        a single model with corresponding pretrained weights.
    """

    if not isinstance(model, str):
        raise ValueError('Invalid input: model should be a string of function name')

    # Setup hub_dir to save downloaded files
    global hub_dir
    if hub_dir is None:
        hub_dir = os.getenv(ENV_TORCH_HUB_DIR, DEFAULT_TORCH_HUB_DIR)

    if '~' in hub_dir:
        hub_dir = os.path.expanduser(hub_dir)

    if not os.path.exists(hub_dir):
        os.makedirs(hub_dir)

    # Parse github repo information
    branch = MASTER_BRANCH
    if ':' in github:
        repo_info, branch = github.split(':')
    else:
        repo_info = github
    repo_owner, repo_name = repo_info.split('/')

    # Download zipped code from github
    url = _git_archive_link(repo_info, branch)
    cached_file = os.path.join(hub_dir, branch + '.zip')
    extracted_repo = os.path.join(hub_dir, repo_name + '-' + branch)
    repo_dir = os.path.join(hub_dir, repo_name + '_' + branch)

    use_cache = (not force_reload) and os.path.exists(repo_dir)

    # Github uses '{repo_name}-{branch_name}' as folder name which is not importable
    # We need to manually rename it to '{repo_name}'
    # Unzip the code and rename the base folder
    if use_cache:
        sys.stderr.write('Using cache found in {}'.format(repo_dir))
    else:
        _remove_if_exists(cached_file)
        _remove_if_exists(extracted_repo)
        _remove_if_exists(repo_dir)

        _download_url_to_file(url, cached_file)
        zipfile.ZipFile(cached_file).extractall(hub_dir)

        _remove_if_exists(cached_file)
        shutil.move(extracted_repo, repo_dir)  # rename the repo

    sys.path.insert(0, repo_dir)  # Make Python interpreter aware of the repo

    dependencies = _load_attr_from_module('hubconf', 'dependencies')

    if dependencies is not None:
        missing_deps = [pkg for pkg in dependencies if not _check_module_exists(pkg)]
        if len(missing_deps):
            raise RuntimeError('Missing dependencies: {}'.format(', '.join(missing_deps)))

    func = _load_attr_from_module('hubconf', model)
    if func is None:
        raise RuntimeError('Cannot find callable {} in hubconf'.format(model))

    # Check if func is callable
    if not callable(func):
        raise RuntimeError('{} is not callable'.format(func))

    # Call the function
    return func(*args, **kwargs)
