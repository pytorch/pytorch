# mypy: allow-untyped-defs
import contextlib
import errno
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import uuid
import warnings
import zipfile
from pathlib import Path
from typing import Any
from typing_extensions import deprecated
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse  # noqa: F401
from urllib.request import Request, urlopen

import torch
from torch.serialization import MAP_LOCATION


class _Faketqdm:  # type: ignore[no-redef]
    def __init__(self, total=None, disable=False, unit=None, *args, **kwargs):
        self.total = total
        self.disable = disable
        self.n = 0
        # Ignore all extra *args and **kwargs lest you want to reinvent tqdm

    def update(self, n):
        if self.disable:
            return

        self.n += n
        if self.total is None:
            sys.stderr.write(f"\r{self.n:.1f} bytes")
        else:
            sys.stderr.write(f"\r{100 * self.n / float(self.total):.1f}%")
        sys.stderr.flush()

    # Don't bother implementing; use real tqdm if you want
    def set_description(self, *args, **kwargs):
        pass

    def write(self, s):
        sys.stderr.write(f"{s}\n")

    def close(self):
        self.disable = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.disable:
            return

        sys.stderr.write("\n")


try:
    from tqdm import tqdm  # If tqdm is installed use it, otherwise use the fake wrapper
except ImportError:
    tqdm = _Faketqdm

__all__ = [
    "download_url_to_file",
    "get_dir",
    "help",
    "list",
    "load",
    "load_state_dict_from_url",
    "set_dir",
]

# matches bfd8deac from resnet18-bfd8deac.pth
HASH_REGEX = re.compile(r"-([a-f0-9]*)\.")

_TRUSTED_REPO_OWNERS = (
    "facebookresearch",
    "facebookincubator",
    "pytorch",
    "fairinternal",
)
ENV_GITHUB_TOKEN = "GITHUB_TOKEN"
ENV_TORCH_HOME = "TORCH_HOME"
ENV_XDG_CACHE_HOME = "XDG_CACHE_HOME"
DEFAULT_CACHE_DIR = "~/.cache"
VAR_DEPENDENCY = "dependencies"
MODULE_HUBCONF = "hubconf.py"
READ_DATA_CHUNK = 128 * 1024
_hub_dir: str | None = None


@contextlib.contextmanager
def _add_to_sys_path(path):
    sys.path.insert(0, path)
    try:
        yield
    finally:
        sys.path.remove(path)


# Copied from tools/shared/module_loader to be included in torch package
def _import_module(name, path):
    import importlib.util
    from importlib.abc import Loader

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None:
        raise AssertionError(f"failed to load spec from {path}")
    module = importlib.util.module_from_spec(spec)
    if not isinstance(spec.loader, Loader):
        raise AssertionError(f"expected Loader, got {type(spec.loader)}")
    spec.loader.exec_module(module)
    return module


def _remove_if_exists(path):
    if os.path.exists(path):
        if os.path.isfile(path):
            os.remove(path)
        else:
            shutil.rmtree(path)


def _git_archive_link(repo_owner, repo_name, ref):
    # See https://docs.github.com/en/rest/reference/repos#download-a-repository-archive-zip
    return f"https://github.com/{repo_owner}/{repo_name}/zipball/{ref}"


def _load_attr_from_module(module, func_name):
    # Check if callable is defined in the module
    if func_name not in dir(module):
        return None
    return getattr(module, func_name)


def _get_torch_home():
    torch_home = os.path.expanduser(
        os.getenv(
            ENV_TORCH_HOME,
            os.path.join(os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR), "torch"),
        )
    )
    return torch_home


def _parse_repo_info(github):
    if ":" in github:
        repo_info, ref = github.split(":")
    else:
        repo_info, ref = github, None
    repo_owner, repo_name = repo_info.split("/")

    if ref is None:
        # The ref wasn't specified by the user, so we need to figure out the
        # default branch: main or master. Our assumption is that if main exists
        # then it's the default branch, otherwise it's master.
        try:
            with urlopen(f"https://github.com/{repo_owner}/{repo_name}/tree/main/"):
                ref = "main"
        except HTTPError as e:
            if e.code == 404:
                ref = "master"
            else:
                raise
        except URLError as e:
            # No internet connection, need to check for cache as last resort
            for possible_ref in ("main", "master"):
                if os.path.exists(
                    f"{get_dir()}/{repo_owner}_{repo_name}_{possible_ref}"
                ):
                    ref = possible_ref
                    break
            if ref is None:
                raise RuntimeError(
                    "It looks like there is no internet connection and the "
                    f"repo could not be found in the cache ({get_dir()})"
                ) from e
    return repo_owner, repo_name, ref


def _read_url(url):
    with urlopen(url) as r:
        return r.read().decode(r.headers.get_content_charset("utf-8"))


def _validate_not_a_forked_repo(repo_owner, repo_name, ref):
    # Use urlopen to avoid depending on local git.
    headers = {"Accept": "application/vnd.github.v3+json"}
    token = os.environ.get(ENV_GITHUB_TOKEN)
    if token is not None:
        headers["Authorization"] = f"token {token}"
    for url_prefix in (
        f"https://api.github.com/repos/{repo_owner}/{repo_name}/branches",
        f"https://api.github.com/repos/{repo_owner}/{repo_name}/tags",
    ):
        page = 0
        while True:
            page += 1
            url = f"{url_prefix}?per_page=100&page={page}"
            try:
                response = json.loads(_read_url(Request(url, headers=headers)))
            except HTTPError:
                # Retry without token in case it had insufficient permissions.
                del headers["Authorization"]
                response = json.loads(_read_url(Request(url, headers=headers)))
            # Empty response means no more data to process
            if not response:
                break
            for br in response:
                if br["name"] == ref or br["commit"]["sha"].startswith(ref):
                    return

    raise ValueError(
        f"Cannot find {ref} in https://github.com/{repo_owner}/{repo_name}. "
        "If it's a commit from a forked repo, please call hub.load() with forked repo directly."
    )


def _get_cache_or_reload(
    github,
    force_reload,
    trust_repo,
    calling_fn,
    verbose=True,
    skip_validation=False,
):
    # Setup hub_dir to save downloaded files
    hub_dir = get_dir()
    os.makedirs(hub_dir, exist_ok=True)
    # Parse github repo information
    repo_owner, repo_name, ref = _parse_repo_info(github)
    # Github allows branch name with slash '/',
    # this causes confusion with path on both Linux and Windows.
    # Backslash is not allowed in Github branch name so no need to
    # to worry about it.
    normalized_br = ref.replace("/", "_")
    # Github renames folder repo-v1.x.x to repo-1.x.x
    # We don't know the repo name before downloading the zip file
    # and inspect name from it.
    # To check if cached repo exists, we need to normalize folder names.
    owner_name_branch = "_".join([repo_owner, repo_name, normalized_br])
    repo_dir = os.path.join(hub_dir, owner_name_branch)
    # Check that the repo is in the trusted list
    _check_repo_is_trusted(
        repo_owner,
        repo_name,
        owner_name_branch,
        trust_repo=trust_repo,
        calling_fn=calling_fn,
    )

    use_cache = (not force_reload) and os.path.exists(repo_dir)

    if use_cache:
        if verbose:
            sys.stderr.write(f"Using cache found in {repo_dir}\n")
    else:
        # Validate the tag/branch is from the original repo instead of a forked repo
        if not skip_validation:
            _validate_not_a_forked_repo(repo_owner, repo_name, ref)

        cached_file = os.path.join(hub_dir, normalized_br + ".zip")
        _remove_if_exists(cached_file)

        try:
            url = _git_archive_link(repo_owner, repo_name, ref)
            sys.stdout.write(f'Downloading: "{url}" to {cached_file}\n')
            download_url_to_file(url, cached_file, progress=False)
        except HTTPError as err:
            if err.code == 300:
                # Getting a 300 Multiple Choices error likely means that the ref is both a tag and a branch
                # in the repo. This can be disambiguated by explicitly using refs/heads/ or refs/tags
                # See https://git-scm.com/book/en/v2/Git-Internals-Git-References
                # Here, we do the same as git: we throw a warning, and assume the user wanted the branch
                warnings.warn(
                    f"The ref {ref} is ambiguous. Perhaps it is both a tag and a branch in the repo? "
                    "Torchhub will now assume that it's a branch. "
                    "You can disambiguate tags and branches by explicitly passing refs/heads/branch_name or "
                    "refs/tags/tag_name as the ref. That might require using skip_validation=True.",
                    stacklevel=2,
                )
                disambiguated_branch_ref = f"refs/heads/{ref}"
                url = _git_archive_link(
                    repo_owner, repo_name, ref=disambiguated_branch_ref
                )
                download_url_to_file(url, cached_file, progress=False)
            else:
                raise

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


def _check_repo_is_trusted(
    repo_owner,
    repo_name,
    owner_name_branch,
    trust_repo,
    calling_fn="load",
):
    hub_dir = get_dir()
    filepath = os.path.join(hub_dir, "trusted_list")

    if not os.path.exists(filepath):
        Path(filepath).touch()
    with open(filepath) as file:
        trusted_repos = tuple(line.strip() for line in file)

    # To minimize friction of introducing the new trust_repo mechanism, we consider that
    # if a repo was already downloaded by torchhub, then it is already trusted (even if it's not in the allowlist)
    trusted_repos_legacy = next(os.walk(hub_dir))[1]

    owner_name = "_".join([repo_owner, repo_name])
    is_trusted = (
        owner_name in trusted_repos
        or owner_name_branch in trusted_repos_legacy
        or repo_owner in _TRUSTED_REPO_OWNERS
    )

    # TODO: Remove `None` option in 2.0 and change the default to "check"
    if trust_repo is None:
        if not is_trusted:
            warnings.warn(
                "You are about to download and run code from an untrusted repository. In a future release, this won't "
                f"be allowed. To add the repository to your trusted list, change the command to {calling_fn}(..., "
                "trust_repo=False) and a command prompt will appear asking for an explicit confirmation of trust, "
                f"or {calling_fn}(..., trust_repo=True), which will assume that the prompt is to be answered with "
                f"'yes'. You can also use {calling_fn}(..., trust_repo='check') which will only prompt for "
                f"confirmation if the repo is not already trusted. This will eventually be the default behaviour",
                stacklevel=2,
            )
        return

    if (trust_repo is False) or (trust_repo == "check" and not is_trusted):
        response = input(
            f"The repository {owner_name} does not belong to the list of trusted repositories and as such cannot be downloaded. "
            "Do you trust this repository and wish to add it to the trusted list of repositories (y/N)?"
        )
        if response.lower() in ("y", "yes"):
            if is_trusted:
                print("The repository is already trusted.")
        elif response.lower() in ("n", "no", ""):
            raise Exception("Untrusted repository.")  # noqa: TRY002
        else:
            raise ValueError(f"Unrecognized response {response}.")

    # At this point we're sure that the user trusts the repo (or wants to trust it)
    if not is_trusted:
        with open(filepath, "a") as file:
            file.write(owner_name + "\n")


def _check_module_exists(name):
    import importlib.util

    return importlib.util.find_spec(name) is not None


def _check_dependencies(m):
    dependencies = _load_attr_from_module(m, VAR_DEPENDENCY)

    if dependencies is not None:
        missing_deps = [pkg for pkg in dependencies if not _check_module_exists(pkg)]
        if missing_deps:
            raise RuntimeError(f"Missing dependencies: {', '.join(missing_deps)}")


def _load_entry_from_hubconf(m, model):
    if not isinstance(model, str):
        raise ValueError("Invalid input: model should be a string of function name")

    # Note that if a missing dependency is imported at top level of hubconf, it will
    # throw before this function. It's a chicken and egg situation where we have to
    # load hubconf to know what're the dependencies, but to import hubconf it requires
    # a missing package. This is fine, Python will throw proper error message for users.
    _check_dependencies(m)

    func = _load_attr_from_module(m, model)

    if func is None or not callable(func):
        raise RuntimeError(f"Cannot find callable {model} in hubconf")

    return func


def get_dir() -> str:
    r"""
    Get the Torch Hub cache directory used for storing downloaded models & weights.

    If :func:`~torch.hub.set_dir` is not called, default path is ``$TORCH_HOME/hub`` where
    environment variable ``$TORCH_HOME`` defaults to ``$XDG_CACHE_HOME/torch``.
    ``$XDG_CACHE_HOME`` follows the X Design Group specification of the Linux
    filesystem layout, with a default value ``~/.cache`` if the environment
    variable is not set.
    """
    # Issue warning to move data if old env is set
    if os.getenv("TORCH_HUB"):
        warnings.warn(
            "TORCH_HUB is deprecated, please use env TORCH_HOME instead", stacklevel=2
        )

    if _hub_dir is not None:
        return _hub_dir
    return os.path.join(_get_torch_home(), "hub")


def set_dir(d: str | os.PathLike) -> None:
    r"""
    Optionally set the Torch Hub directory used to save downloaded models & weights.

    Args:
        d (str): path to a local folder to save downloaded models & weights.
    """
    global _hub_dir
    _hub_dir = os.path.expanduser(d)


def list(
    github,
    force_reload=False,
    skip_validation=False,
    trust_repo=None,
    verbose=True,
):
    r"""
    List all callable entrypoints available in the repo specified by ``github``.

    Args:
        github (str): a string with format "repo_owner/repo_name[:ref]" with an optional
            ref (tag or branch). If ``ref`` is not specified, the default branch is assumed to be ``main`` if
            it exists, and otherwise ``master``.
            Example: 'pytorch/vision:0.10'
        force_reload (bool, optional): whether to discard the existing cache and force a fresh download.
            Default is ``False``.
        skip_validation (bool, optional): if ``False``, torchhub will check that the branch or commit
            specified by the ``github`` argument properly belongs to the repo owner. This will make
            requests to the GitHub API; you can specify a non-default GitHub token by setting the
            ``GITHUB_TOKEN`` environment variable. Default is ``False``.
        trust_repo (bool, str or None): ``"check"``, ``True``, ``False`` or ``None``.
            This parameter was introduced in v1.12 and helps ensuring that users
            only run code from repos that they trust.

            - If ``False``, a prompt will ask the user whether the repo should
              be trusted.
            - If ``True``, the repo will be added to the trusted list and loaded
              without requiring explicit confirmation.
            - If ``"check"``, the repo will be checked against the list of
              trusted repos in the cache. If it is not present in that list, the
              behaviour will fall back onto the ``trust_repo=False`` option.
            - If ``None``: this will raise a warning, inviting the user to set
              ``trust_repo`` to either ``False``, ``True`` or ``"check"``. This
              is only present for backward compatibility and will be removed in
              v2.0.

            Default is ``None`` and will eventually change to ``"check"`` in v2.0.
        verbose (bool, optional): If ``False``, mute messages about hitting
            local caches. Note that the message about first download cannot be
            muted. Default is ``True``.

    Returns:
        list: The available callables entrypoint

    Example:
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_HUB)
        >>> entrypoints = torch.hub.list("pytorch/vision", force_reload=True)
    """
    repo_dir = _get_cache_or_reload(
        github,
        force_reload,
        trust_repo,
        "list",
        verbose=verbose,
        skip_validation=skip_validation,
    )

    with _add_to_sys_path(repo_dir):
        hubconf_path = os.path.join(repo_dir, MODULE_HUBCONF)
        hub_module = _import_module(MODULE_HUBCONF, hubconf_path)

    # We take functions starts with '_' as internal helper functions
    entrypoints = [
        f
        for f in dir(hub_module)
        if callable(getattr(hub_module, f)) and not f.startswith("_")
    ]

    return entrypoints


def help(github, model, force_reload=False, skip_validation=False, trust_repo=None):
    r"""
    Show the docstring of entrypoint ``model``.

    Args:
        github (str): a string with format <repo_owner/repo_name[:ref]> with an optional
            ref (a tag or a branch). If ``ref`` is not specified, the default branch is assumed
            to be ``main`` if it exists, and otherwise ``master``.
            Example: 'pytorch/vision:0.10'
        model (str): a string of entrypoint name defined in repo's ``hubconf.py``
        force_reload (bool, optional): whether to discard the existing cache and force a fresh download.
            Default is ``False``.
        skip_validation (bool, optional): if ``False``, torchhub will check that the ref
            specified by the ``github`` argument properly belongs to the repo owner. This will make
            requests to the GitHub API; you can specify a non-default GitHub token by setting the
            ``GITHUB_TOKEN`` environment variable. Default is ``False``.
        trust_repo (bool, str or None): ``"check"``, ``True``, ``False`` or ``None``.
            This parameter was introduced in v1.12 and helps ensuring that users
            only run code from repos that they trust.

            - If ``False``, a prompt will ask the user whether the repo should
              be trusted.
            - If ``True``, the repo will be added to the trusted list and loaded
              without requiring explicit confirmation.
            - If ``"check"``, the repo will be checked against the list of
              trusted repos in the cache. If it is not present in that list, the
              behaviour will fall back onto the ``trust_repo=False`` option.
            - If ``None``: this will raise a warning, inviting the user to set
              ``trust_repo`` to either ``False``, ``True`` or ``"check"``. This
              is only present for backward compatibility and will be removed in
              v2.0.

            Default is ``None`` and will eventually change to ``"check"`` in v2.0.
    Example:
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_HUB)
        >>> print(torch.hub.help("pytorch/vision", "resnet18", force_reload=True))
    """
    repo_dir = _get_cache_or_reload(
        github,
        force_reload,
        trust_repo,
        "help",
        verbose=True,
        skip_validation=skip_validation,
    )

    with _add_to_sys_path(repo_dir):
        hubconf_path = os.path.join(repo_dir, MODULE_HUBCONF)
        hub_module = _import_module(MODULE_HUBCONF, hubconf_path)

    entry = _load_entry_from_hubconf(hub_module, model)

    return entry.__doc__


def load(
    repo_or_dir,
    model,
    *args,
    source="github",
    trust_repo=None,
    force_reload=False,
    verbose=True,
    skip_validation=False,
    **kwargs,
):
    r"""
    Load a model from a github repo or a local directory.

    Note: Loading a model is the typical use case, but this can also be used to
    for loading other objects such as tokenizers, loss functions, etc.

    If ``source`` is 'github', ``repo_or_dir`` is expected to be
    of the form ``repo_owner/repo_name[:ref]`` with an optional
    ref (a tag or a branch).

    If ``source`` is 'local', ``repo_or_dir`` is expected to be a
    path to a local directory.

    Args:
        repo_or_dir (str): If ``source`` is 'github',
            this should correspond to a github repo with format ``repo_owner/repo_name[:ref]`` with
            an optional ref (tag or branch), for example 'pytorch/vision:0.10'. If ``ref`` is not specified,
            the default branch is assumed to be ``main`` if it exists, and otherwise ``master``.
            If ``source`` is 'local'  then it should be a path to a local directory.
        model (str): the name of a callable (entrypoint) defined in the
            repo/dir's ``hubconf.py``.
        *args (optional): the corresponding args for callable ``model``.
        source (str, optional): 'github' or 'local'. Specifies how
            ``repo_or_dir`` is to be interpreted. Default is 'github'.
        trust_repo (bool, str or None): ``"check"``, ``True``, ``False`` or ``None``.
            This parameter was introduced in v1.12 and helps ensuring that users
            only run code from repos that they trust.

            - If ``False``, a prompt will ask the user whether the repo should
              be trusted.
            - If ``True``, the repo will be added to the trusted list and loaded
              without requiring explicit confirmation.
            - If ``"check"``, the repo will be checked against the list of
              trusted repos in the cache. If it is not present in that list, the
              behaviour will fall back onto the ``trust_repo=False`` option.
            - If ``None``: this will raise a warning, inviting the user to set
              ``trust_repo`` to either ``False``, ``True`` or ``"check"``. This
              is only present for backward compatibility and will be removed in
              v2.0.

            Default is ``None`` and will eventually change to ``"check"`` in v2.0.
        force_reload (bool, optional): whether to force a fresh download of
            the github repo unconditionally. Does not have any effect if
            ``source = 'local'``. Default is ``False``.
        verbose (bool, optional): If ``False``, mute messages about hitting
            local caches. Note that the message about first download cannot be
            muted. Does not have any effect if ``source = 'local'``.
            Default is ``True``.
        skip_validation (bool, optional): if ``False``, torchhub will check that the branch or commit
            specified by the ``github`` argument properly belongs to the repo owner. This will make
            requests to the GitHub API; you can specify a non-default GitHub token by setting the
            ``GITHUB_TOKEN`` environment variable. Default is ``False``.
        **kwargs (optional): the corresponding kwargs for callable ``model``.

    Returns:
        The output of the ``model`` callable when called with the given
        ``*args`` and ``**kwargs``.

    Example:
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_HUB)
        >>> # from a github repo
        >>> repo = "pytorch/vision"
        >>> model = torch.hub.load(
        ...     repo, "resnet50", weights="ResNet50_Weights.IMAGENET1K_V1"
        ... )
        >>> # from a local directory
        >>> path = "/some/local/path/pytorch/vision"
        >>> # xdoctest: +SKIP
        >>> model = torch.hub.load(path, "resnet50", weights="ResNet50_Weights.DEFAULT")
    """
    source = source.lower()

    if source not in ("github", "local"):
        raise ValueError(
            f'Unknown source: "{source}". Allowed values: "github" | "local".'
        )

    if source == "github":
        repo_or_dir = _get_cache_or_reload(
            repo_or_dir,
            force_reload,
            trust_repo,
            "load",
            verbose=verbose,
            skip_validation=skip_validation,
        )

    model = _load_local(repo_or_dir, model, *args, **kwargs)
    return model


def _load_local(hubconf_dir, model, *args, **kwargs):
    r"""
    Load a model from a local directory with a ``hubconf.py``.

    Args:
        hubconf_dir (str): path to a local directory that contains a
            ``hubconf.py``.
        model (str): name of an entrypoint defined in the directory's
            ``hubconf.py``.
        *args (optional): the corresponding args for callable ``model``.
        **kwargs (optional): the corresponding kwargs for callable ``model``.

    Returns:
        a single model with corresponding pretrained weights.

    Example:
        >>> # xdoctest: +SKIP("stub local path")
        >>> path = "/some/local/path/pytorch/vision"
        >>> model = _load_local(
        ...     path,
        ...     "resnet50",
        ...     weights="ResNet50_Weights.IMAGENET1K_V1",
        ... )
    """
    with _add_to_sys_path(hubconf_dir):
        hubconf_path = os.path.join(hubconf_dir, MODULE_HUBCONF)
        hub_module = _import_module(MODULE_HUBCONF, hubconf_path)

        entry = _load_entry_from_hubconf(hub_module, model)
        model = entry(*args, **kwargs)

    return model


def download_url_to_file(
    url: str,
    dst: str,
    hash_prefix: str | None = None,
    progress: bool = True,
) -> None:
    r"""Download object at the given URL to a local path.

    Args:
        url (str): URL of the object to download
        dst (str): Full path where object will be saved, e.g. ``/tmp/temporary_file``
        hash_prefix (str, optional): If not None, the SHA256 downloaded file should start with ``hash_prefix``.
            Default: None
        progress (bool, optional): whether or not to display a progress bar to stderr
            Default: True

    Example:
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_HUB)
        >>> # xdoctest: +REQUIRES(POSIX)
        >>> torch.hub.download_url_to_file(
        ...     "https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth",
        ...     "/tmp/temporary_file",
        ... )

    """
    # We deliberately save it in a temp file and move it after
    # download is complete. This prevents a local working checkpoint
    # being overridden by a broken download.
    # We deliberately do not use NamedTemporaryFile to avoid restrictive
    # file permissions being applied to the downloaded file.
    dst = os.path.expanduser(dst)
    for _ in range(tempfile.TMP_MAX):
        tmp_dst = dst + "." + uuid.uuid4().hex + ".partial"
        try:
            f = open(tmp_dst, "w+b")  # noqa: SIM115
        except FileExistsError:
            continue
        break
    else:
        raise FileExistsError(errno.EEXIST, "No usable temporary file name found")
    req = Request(url, headers={"User-Agent": "torch.hub"})
    try:
        with urlopen(req) as u:
            meta = u.info()
            if hasattr(meta, "getheaders"):
                content_length = meta.getheaders("Content-Length")
            else:
                content_length = meta.get_all("Content-Length")
            file_size = None
            if content_length is not None and len(content_length) > 0:
                file_size = int(content_length[0])

            sha256 = hashlib.sha256() if hash_prefix is not None else None
            with tqdm(
                total=file_size,
                disable=not progress,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                while True:
                    buffer = u.read(READ_DATA_CHUNK)
                    if len(buffer) == 0:
                        break
                    f.write(buffer)
                    if sha256 is not None:
                        sha256.update(buffer)
                    pbar.update(len(buffer))

            f.close()
            if sha256 is not None and hash_prefix is not None:
                digest = sha256.hexdigest()
                if digest[: len(hash_prefix)] != hash_prefix:
                    raise RuntimeError(
                        f'invalid hash value (expected "{hash_prefix}", got "{digest}")'
                    )
        shutil.move(f.name, dst)
    finally:
        f.close()
        if os.path.exists(f.name):
            os.remove(f.name)


# Hub used to support automatically extracts from zipfile manually compressed by users.
# The legacy zip format expects only one file from torch.save() < 1.6 in the zip.
# We should remove this support since zipfile is now default zipfile format for torch.save().
def _is_legacy_zip_format(filename: str) -> bool:
    if zipfile.is_zipfile(filename):
        with zipfile.ZipFile(filename) as zf:
            infolist = zf.infolist()
        return len(infolist) == 1 and not infolist[0].is_dir()
    return False


@deprecated(
    "Falling back to the old format < 1.6. This support will be "
    "deprecated in favor of default zipfile format introduced in 1.6. "
    "Please redo torch.save() to save it in the new zipfile format.",
    category=FutureWarning,
)
def _legacy_zip_load(
    filename: str,
    model_dir: str,
    map_location: MAP_LOCATION,
    weights_only: bool,
) -> dict[str, Any]:
    # Note: extractall() defaults to overwrite file if exists. No need to clean up beforehand.
    #       We deliberately don't handle tarfile here since our legacy serialization format was in tar.
    #       E.g. resnet18-5c106cde.pth which is widely used.
    with zipfile.ZipFile(filename) as f:
        members = f.infolist()
        if len(members) != 1:
            raise RuntimeError("Only one file(not dir) is allowed in the zipfile")
        f.extractall(model_dir)
        extraced_name = members[0].filename
        extracted_file = os.path.join(model_dir, extraced_name)
    return torch.load(
        extracted_file, map_location=map_location, weights_only=weights_only
    )


def load_state_dict_from_url(
    url: str,
    model_dir: str | None = None,
    map_location: MAP_LOCATION = None,
    progress: bool = True,
    check_hash: bool = False,
    file_name: str | None = None,
    weights_only: bool = False,
) -> dict[str, Any]:
    r"""Loads the Torch serialized object at the given URL.

    If downloaded file is a zip file, it will be automatically
    decompressed.

    If the object is already present in `model_dir`, it's deserialized and
    returned.
    The default value of ``model_dir`` is ``<hub_dir>/checkpoints`` where
    ``hub_dir`` is the directory returned by :func:`~torch.hub.get_dir`.

    Args:
        url (str): URL of the object to download
        model_dir (str, optional): directory in which to save the object
        map_location (optional): a function or a dict specifying how to remap storage locations (see torch.load)
        progress (bool, optional): whether or not to display a progress bar to stderr.
            Default: True
        check_hash(bool, optional): If True, the filename part of the URL should follow the naming convention
            ``filename-<sha256>.ext`` where ``<sha256>`` is the first eight or more
            digits of the SHA256 hash of the contents of the file. The hash is used to
            ensure unique names and to verify the contents of the file.
            Default: False
        file_name (str, optional): name for the downloaded file. Filename from ``url`` will be used if not set.
        weights_only(bool, optional): If True, only weights will be loaded and no complex pickled objects.
            Recommended for untrusted sources. See :func:`~torch.load` for more details.

    Example:
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_HUB)
        >>> state_dict = torch.hub.load_state_dict_from_url(
        ...     "https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth"
        ... )

    """
    # Issue warning to move data if old env is set
    if os.getenv("TORCH_MODEL_ZOO"):
        warnings.warn(
            "TORCH_MODEL_ZOO is deprecated, please use env TORCH_HOME instead",
            stacklevel=2,
        )

    if model_dir is None:
        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, "checkpoints")

    os.makedirs(model_dir, exist_ok=True)

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stdout.write(f'Downloading: "{url}" to {cached_file}\n')
        hash_prefix = None
        if check_hash:
            r = HASH_REGEX.search(filename)  # r is Optional[Match[str]]
            hash_prefix = r.group(1) if r else None
        download_url_to_file(url, cached_file, hash_prefix, progress=progress)

    if _is_legacy_zip_format(cached_file):
        return _legacy_zip_load(cached_file, model_dir, map_location, weights_only)
    return torch.load(cached_file, map_location=map_location, weights_only=weights_only)
