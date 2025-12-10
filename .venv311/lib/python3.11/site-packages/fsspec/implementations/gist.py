import requests

from ..spec import AbstractFileSystem
from ..utils import infer_storage_options
from .memory import MemoryFile


class GistFileSystem(AbstractFileSystem):
    """
    Interface to files in a single GitHub Gist.

    Provides read-only access to a gist's files. Gists do not contain
    subdirectories, so file listing is straightforward.

    Parameters
    ----------
    gist_id: str
        The ID of the gist you want to access (the long hex value from the URL).
    filenames: list[str] (optional)
        If provided, only make a file system representing these files, and do not fetch
        the list of all files for this gist.
    sha: str (optional)
        If provided, fetch a particular revision of the gist. If omitted,
        the latest revision is used.
    username: str (optional)
        GitHub username for authentication.
    token: str (optional)
        GitHub personal access token (required if username is given), or.
    timeout: (float, float) or float, optional
        Connect and read timeouts for requests (default 60s each).
    kwargs: dict
        Stored on `self.request_kw` and passed to `requests.get` when fetching Gist
        metadata or reading ("opening") a file.
    """

    protocol = "gist"
    gist_url = "https://api.github.com/gists/{gist_id}"
    gist_rev_url = "https://api.github.com/gists/{gist_id}/{sha}"

    def __init__(
        self,
        gist_id,
        filenames=None,
        sha=None,
        username=None,
        token=None,
        timeout=None,
        **kwargs,
    ):
        super().__init__()
        self.gist_id = gist_id
        self.filenames = filenames
        self.sha = sha  # revision of the gist (optional)
        if username is not None and token is None:
            raise ValueError("User auth requires a token")
        self.username = username
        self.token = token
        self.request_kw = kwargs
        # Default timeouts to 60s connect/read if none provided
        self.timeout = timeout if timeout is not None else (60, 60)

        # We use a single-level "directory" cache, because a gist is essentially flat
        self.dircache[""] = self._fetch_file_list()

    @property
    def kw(self):
        """Auth parameters passed to 'requests' if we have username/token."""
        kw = {
            "headers": {
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            }
        }
        kw.update(self.request_kw)
        if self.username and self.token:
            kw["auth"] = (self.username, self.token)
        elif self.token:
            kw["headers"]["Authorization"] = f"Bearer {self.token}"
        return kw

    def _fetch_gist_metadata(self):
        """
        Fetch the JSON metadata for this gist (possibly for a specific revision).
        """
        if self.sha:
            url = self.gist_rev_url.format(gist_id=self.gist_id, sha=self.sha)
        else:
            url = self.gist_url.format(gist_id=self.gist_id)

        r = requests.get(url, timeout=self.timeout, **self.kw)
        if r.status_code == 404:
            raise FileNotFoundError(
                f"Gist not found: {self.gist_id}@{self.sha or 'latest'}"
            )
        r.raise_for_status()
        return r.json()

    def _fetch_file_list(self):
        """
        Returns a list of dicts describing each file in the gist. These get stored
        in self.dircache[""].
        """
        meta = self._fetch_gist_metadata()
        if self.filenames:
            available_files = meta.get("files", {})
            files = {}
            for fn in self.filenames:
                if fn not in available_files:
                    raise FileNotFoundError(fn)
                files[fn] = available_files[fn]
        else:
            files = meta.get("files", {})

        out = []
        for fname, finfo in files.items():
            if finfo is None:
                # Occasionally GitHub returns a file entry with null if it was deleted
                continue
            # Build a directory entry
            out.append(
                {
                    "name": fname,  # file's name
                    "type": "file",  # gists have no subdirectories
                    "size": finfo.get("size", 0),  # file size in bytes
                    "raw_url": finfo.get("raw_url"),
                }
            )
        return out

    @classmethod
    def _strip_protocol(cls, path):
        """
        Remove 'gist://' from the path, if present.
        """
        # The default infer_storage_options can handle gist://username:token@id/file
        # or gist://id/file, but let's ensure we handle a normal usage too.
        # We'll just strip the protocol prefix if it exists.
        path = infer_storage_options(path).get("path", path)
        return path.lstrip("/")

    @staticmethod
    def _get_kwargs_from_urls(path):
        """
        Parse 'gist://' style URLs into GistFileSystem constructor kwargs.
        For example:
          gist://:TOKEN@<gist_id>/file.txt
          gist://username:TOKEN@<gist_id>/file.txt
        """
        so = infer_storage_options(path)
        out = {}
        if "username" in so and so["username"]:
            out["username"] = so["username"]
        if "password" in so and so["password"]:
            out["token"] = so["password"]
        if "host" in so and so["host"]:
            # We interpret 'host' as the gist ID
            out["gist_id"] = so["host"]

        # Extract SHA and filename from path
        if "path" in so and so["path"]:
            path_parts = so["path"].rsplit("/", 2)[-2:]
            if len(path_parts) == 2:
                if path_parts[0]:  # SHA present
                    out["sha"] = path_parts[0]
                if path_parts[1]:  # filename also present
                    out["filenames"] = [path_parts[1]]

        return out

    def ls(self, path="", detail=False, **kwargs):
        """
        List files in the gist. Gists are single-level, so any 'path' is basically
        the filename, or empty for all files.

        Parameters
        ----------
        path : str, optional
            The filename to list. If empty, returns all files in the gist.
        detail : bool, default False
            If True, return a list of dicts; if False, return a list of filenames.
        """
        path = self._strip_protocol(path or "")
        # If path is empty, return all
        if path == "":
            results = self.dircache[""]
        else:
            # We want just the single file with this name
            all_files = self.dircache[""]
            results = [f for f in all_files if f["name"] == path]
            if not results:
                raise FileNotFoundError(path)
        if detail:
            return results
        else:
            return sorted(f["name"] for f in results)

    def _open(self, path, mode="rb", block_size=None, **kwargs):
        """
        Read a single file from the gist.
        """
        if mode != "rb":
            raise NotImplementedError("GitHub Gist FS is read-only (no write).")

        path = self._strip_protocol(path)
        # Find the file entry in our dircache
        matches = [f for f in self.dircache[""] if f["name"] == path]
        if not matches:
            raise FileNotFoundError(path)
        finfo = matches[0]

        raw_url = finfo.get("raw_url")
        if not raw_url:
            raise FileNotFoundError(f"No raw_url for file: {path}")

        r = requests.get(raw_url, timeout=self.timeout, **self.kw)
        if r.status_code == 404:
            raise FileNotFoundError(path)
        r.raise_for_status()
        return MemoryFile(path, None, r.content)

    def cat(self, path, recursive=False, on_error="raise", **kwargs):
        """
        Return {path: contents} for the given file or files. If 'recursive' is True,
        and path is empty, returns all files in the gist.
        """
        paths = self.expand_path(path, recursive=recursive)
        out = {}
        for p in paths:
            try:
                with self.open(p, "rb") as f:
                    out[p] = f.read()
            except FileNotFoundError as e:
                if on_error == "raise":
                    raise e
                elif on_error == "omit":
                    pass  # skip
                else:
                    out[p] = e
        if len(paths) == 1 and paths[0] == path:
            return out[path]
        return out
