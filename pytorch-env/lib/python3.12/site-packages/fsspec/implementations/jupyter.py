import base64
import io
import re

import requests

import fsspec


class JupyterFileSystem(fsspec.AbstractFileSystem):
    """View of the files as seen by a Jupyter server (notebook or lab)"""

    protocol = ("jupyter", "jlab")

    def __init__(self, url, tok=None, **kwargs):
        """

        Parameters
        ----------
        url : str
            Base URL of the server, like "http://127.0.0.1:8888". May include
            token in the string, which is given by the process when starting up
        tok : str
            If the token is obtained separately, can be given here
        kwargs
        """
        if "?" in url:
            if tok is None:
                try:
                    tok = re.findall("token=([a-z0-9]+)", url)[0]
                except IndexError as e:
                    raise ValueError("Could not determine token") from e
            url = url.split("?", 1)[0]
        self.url = url.rstrip("/") + "/api/contents"
        self.session = requests.Session()
        if tok:
            self.session.headers["Authorization"] = f"token {tok}"

        super().__init__(**kwargs)

    def ls(self, path, detail=True, **kwargs):
        path = self._strip_protocol(path)
        r = self.session.get(f"{self.url}/{path}")
        if r.status_code == 404:
            return FileNotFoundError(path)
        r.raise_for_status()
        out = r.json()

        if out["type"] == "directory":
            out = out["content"]
        else:
            out = [out]
        for o in out:
            o["name"] = o.pop("path")
            o.pop("content")
            if o["type"] == "notebook":
                o["type"] = "file"
        if detail:
            return out
        return [o["name"] for o in out]

    def cat_file(self, path, start=None, end=None, **kwargs):
        path = self._strip_protocol(path)
        r = self.session.get(f"{self.url}/{path}")
        if r.status_code == 404:
            return FileNotFoundError(path)
        r.raise_for_status()
        out = r.json()
        if out["format"] == "text":
            # data should be binary
            b = out["content"].encode()
        else:
            b = base64.b64decode(out["content"])
        return b[start:end]

    def pipe_file(self, path, value, **_):
        path = self._strip_protocol(path)
        json = {
            "name": path.rsplit("/", 1)[-1],
            "path": path,
            "size": len(value),
            "content": base64.b64encode(value).decode(),
            "format": "base64",
            "type": "file",
        }
        self.session.put(f"{self.url}/{path}", json=json)

    def mkdir(self, path, create_parents=True, **kwargs):
        path = self._strip_protocol(path)
        if create_parents and "/" in path:
            self.mkdir(path.rsplit("/", 1)[0], True)
        json = {
            "name": path.rsplit("/", 1)[-1],
            "path": path,
            "size": None,
            "content": None,
            "type": "directory",
        }
        self.session.put(f"{self.url}/{path}", json=json)

    def _rm(self, path):
        path = self._strip_protocol(path)
        self.session.delete(f"{self.url}/{path}")

    def _open(self, path, mode="rb", **kwargs):
        path = self._strip_protocol(path)
        if mode == "rb":
            data = self.cat_file(path)
            return io.BytesIO(data)
        else:
            return SimpleFileWriter(self, path, mode="wb")


class SimpleFileWriter(fsspec.spec.AbstractBufferedFile):
    def _upload_chunk(self, final=False):
        """Never uploads a chunk until file is done

        Not suitable for large files
        """
        if final is False:
            return False
        self.buffer.seek(0)
        data = self.buffer.read()
        self.fs.pipe_file(self.path, data)
