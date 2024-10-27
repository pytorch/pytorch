import requests

import fsspec

from ..spec import AbstractFileSystem
from ..utils import infer_storage_options
from .memory import MemoryFile

# TODO: add GIST backend, would be very similar


class GithubFileSystem(AbstractFileSystem):
    """Interface to files in github

    An instance of this class provides the files residing within a remote github
    repository. You may specify a point in the repos history, by SHA, branch
    or tag (default is current master).

    Given that code files tend to be small, and that github does not support
    retrieving partial content, we always fetch whole files.

    When using fsspec.open, allows URIs of the form:

    - "github://path/file", in which case you must specify org, repo and
      may specify sha in the extra args
    - 'github://org:repo@/precip/catalog.yml', where the org and repo are
      part of the URI
    - 'github://org:repo@sha/precip/catalog.yml', where the sha is also included

    ``sha`` can be the full or abbreviated hex of the commit you want to fetch
    from, or a branch or tag name (so long as it doesn't contain special characters
    like "/", "?", which would have to be HTTP-encoded).

    For authorised access, you must provide username and token, which can be made
    at https://github.com/settings/tokens
    """

    url = "https://api.github.com/repos/{org}/{repo}/git/trees/{sha}"
    rurl = "https://raw.githubusercontent.com/{org}/{repo}/{sha}/{path}"
    protocol = "github"
    timeout = (60, 60)  # connect, read timeouts

    def __init__(
        self, org, repo, sha=None, username=None, token=None, timeout=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.org = org
        self.repo = repo
        if (username is None) ^ (token is None):
            raise ValueError("Auth required both username and token")
        self.username = username
        self.token = token
        if timeout is not None:
            self.timeout = timeout
        if sha is None:
            # look up default branch (not necessarily "master")
            u = "https://api.github.com/repos/{org}/{repo}"
            r = requests.get(
                u.format(org=org, repo=repo), timeout=self.timeout, **self.kw
            )
            r.raise_for_status()
            sha = r.json()["default_branch"]

        self.root = sha
        self.ls("")

    @property
    def kw(self):
        if self.username:
            return {"auth": (self.username, self.token)}
        return {}

    @classmethod
    def repos(cls, org_or_user, is_org=True):
        """List repo names for given org or user

        This may become the top level of the FS

        Parameters
        ----------
        org_or_user: str
            Name of the github org or user to query
        is_org: bool (default True)
            Whether the name is an organisation (True) or user (False)

        Returns
        -------
        List of string
        """
        r = requests.get(
            f"https://api.github.com/{['users', 'orgs'][is_org]}/{org_or_user}/repos",
            timeout=cls.timeout,
        )
        r.raise_for_status()
        return [repo["name"] for repo in r.json()]

    @property
    def tags(self):
        """Names of tags in the repo"""
        r = requests.get(
            f"https://api.github.com/repos/{self.org}/{self.repo}/tags",
            timeout=self.timeout,
            **self.kw,
        )
        r.raise_for_status()
        return [t["name"] for t in r.json()]

    @property
    def branches(self):
        """Names of branches in the repo"""
        r = requests.get(
            f"https://api.github.com/repos/{self.org}/{self.repo}/branches",
            timeout=self.timeout,
            **self.kw,
        )
        r.raise_for_status()
        return [t["name"] for t in r.json()]

    @property
    def refs(self):
        """Named references, tags and branches"""
        return {"tags": self.tags, "branches": self.branches}

    def ls(self, path, detail=False, sha=None, _sha=None, **kwargs):
        """List files at given path

        Parameters
        ----------
        path: str
            Location to list, relative to repo root
        detail: bool
            If True, returns list of dicts, one per file; if False, returns
            list of full filenames only
        sha: str (optional)
            List at the given point in the repo history, branch or tag name or commit
            SHA
        _sha: str (optional)
            List this specific tree object (used internally to descend into trees)
        """
        path = self._strip_protocol(path)
        if path == "":
            _sha = sha or self.root
        if _sha is None:
            parts = path.rstrip("/").split("/")
            so_far = ""
            _sha = sha or self.root
            for part in parts:
                out = self.ls(so_far, True, sha=sha, _sha=_sha)
                so_far += "/" + part if so_far else part
                out = [o for o in out if o["name"] == so_far]
                if not out:
                    raise FileNotFoundError(path)
                out = out[0]
                if out["type"] == "file":
                    if detail:
                        return [out]
                    else:
                        return path
                _sha = out["sha"]
        if path not in self.dircache or sha not in [self.root, None]:
            r = requests.get(
                self.url.format(org=self.org, repo=self.repo, sha=_sha),
                timeout=self.timeout,
                **self.kw,
            )
            if r.status_code == 404:
                raise FileNotFoundError(path)
            r.raise_for_status()
            types = {"blob": "file", "tree": "directory"}
            out = [
                {
                    "name": path + "/" + f["path"] if path else f["path"],
                    "mode": f["mode"],
                    "type": types[f["type"]],
                    "size": f.get("size", 0),
                    "sha": f["sha"],
                }
                for f in r.json()["tree"]
                if f["type"] in types
            ]
            if sha in [self.root, None]:
                self.dircache[path] = out
        else:
            out = self.dircache[path]
        if detail:
            return out
        else:
            return sorted([f["name"] for f in out])

    def invalidate_cache(self, path=None):
        self.dircache.clear()

    @classmethod
    def _strip_protocol(cls, path):
        opts = infer_storage_options(path)
        if "username" not in opts:
            return super()._strip_protocol(path)
        return opts["path"].lstrip("/")

    @staticmethod
    def _get_kwargs_from_urls(path):
        opts = infer_storage_options(path)
        if "username" not in opts:
            return {}
        out = {"org": opts["username"], "repo": opts["password"]}
        if opts["host"]:
            out["sha"] = opts["host"]
        return out

    def _open(
        self,
        path,
        mode="rb",
        block_size=None,
        autocommit=True,
        cache_options=None,
        sha=None,
        **kwargs,
    ):
        if mode != "rb":
            raise NotImplementedError
        url = self.rurl.format(
            org=self.org, repo=self.repo, path=path, sha=sha or self.root
        )
        r = requests.get(url, timeout=self.timeout, **self.kw)
        if r.status_code == 404:
            raise FileNotFoundError(path)
        r.raise_for_status()
        return MemoryFile(None, None, r.content)

    def cat(self, path, recursive=False, on_error="raise", **kwargs):
        paths = self.expand_path(path, recursive=recursive)
        urls = [
            self.rurl.format(org=self.org, repo=self.repo, path=u, sha=self.root)
            for u, sh in paths
        ]
        fs = fsspec.filesystem("http")
        data = fs.cat(urls, on_error="return")
        return {u: v for ((k, v), u) in zip(data.items(), urls)}
