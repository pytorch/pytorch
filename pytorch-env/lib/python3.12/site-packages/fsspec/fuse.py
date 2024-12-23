import argparse
import logging
import os
import stat
import threading
import time
from errno import EIO, ENOENT

from fuse import FUSE, FuseOSError, LoggingMixIn, Operations

from fsspec import __version__
from fsspec.core import url_to_fs

logger = logging.getLogger("fsspec.fuse")


class FUSEr(Operations):
    def __init__(self, fs, path, ready_file=False):
        self.fs = fs
        self.cache = {}
        self.root = path.rstrip("/") + "/"
        self.counter = 0
        logger.info("Starting FUSE at %s", path)
        self._ready_file = ready_file

    def getattr(self, path, fh=None):
        logger.debug("getattr %s", path)
        if self._ready_file and path in ["/.fuse_ready", ".fuse_ready"]:
            return {"type": "file", "st_size": 5}

        path = "".join([self.root, path.lstrip("/")]).rstrip("/")
        try:
            info = self.fs.info(path)
        except FileNotFoundError as exc:
            raise FuseOSError(ENOENT) from exc

        data = {"st_uid": info.get("uid", 1000), "st_gid": info.get("gid", 1000)}
        perm = info.get("mode", 0o777)

        if info["type"] != "file":
            data["st_mode"] = stat.S_IFDIR | perm
            data["st_size"] = 0
            data["st_blksize"] = 0
        else:
            data["st_mode"] = stat.S_IFREG | perm
            data["st_size"] = info["size"]
            data["st_blksize"] = 5 * 2**20
            data["st_nlink"] = 1
        data["st_atime"] = info["atime"] if "atime" in info else time.time()
        data["st_ctime"] = info["ctime"] if "ctime" in info else time.time()
        data["st_mtime"] = info["mtime"] if "mtime" in info else time.time()
        return data

    def readdir(self, path, fh):
        logger.debug("readdir %s", path)
        path = "".join([self.root, path.lstrip("/")])
        files = self.fs.ls(path, False)
        files = [os.path.basename(f.rstrip("/")) for f in files]
        return [".", ".."] + files

    def mkdir(self, path, mode):
        path = "".join([self.root, path.lstrip("/")])
        self.fs.mkdir(path)
        return 0

    def rmdir(self, path):
        path = "".join([self.root, path.lstrip("/")])
        self.fs.rmdir(path)
        return 0

    def read(self, path, size, offset, fh):
        logger.debug("read %s", (path, size, offset))
        if self._ready_file and path in ["/.fuse_ready", ".fuse_ready"]:
            # status indicator
            return b"ready"

        f = self.cache[fh]
        f.seek(offset)
        out = f.read(size)
        return out

    def write(self, path, data, offset, fh):
        logger.debug("write %s", (path, offset))
        f = self.cache[fh]
        f.seek(offset)
        f.write(data)
        return len(data)

    def create(self, path, flags, fi=None):
        logger.debug("create %s", (path, flags))
        fn = "".join([self.root, path.lstrip("/")])
        self.fs.touch(fn)  # OS will want to get attributes immediately
        f = self.fs.open(fn, "wb")
        self.cache[self.counter] = f
        self.counter += 1
        return self.counter - 1

    def open(self, path, flags):
        logger.debug("open %s", (path, flags))
        fn = "".join([self.root, path.lstrip("/")])
        if flags % 2 == 0:
            # read
            mode = "rb"
        else:
            # write/create
            mode = "wb"
        self.cache[self.counter] = self.fs.open(fn, mode)
        self.counter += 1
        return self.counter - 1

    def truncate(self, path, length, fh=None):
        fn = "".join([self.root, path.lstrip("/")])
        if length != 0:
            raise NotImplementedError
        # maybe should be no-op since open with write sets size to zero anyway
        self.fs.touch(fn)

    def unlink(self, path):
        fn = "".join([self.root, path.lstrip("/")])
        try:
            self.fs.rm(fn, False)
        except (OSError, FileNotFoundError) as exc:
            raise FuseOSError(EIO) from exc

    def release(self, path, fh):
        try:
            if fh in self.cache:
                f = self.cache[fh]
                f.close()
                self.cache.pop(fh)
        except Exception as e:
            print(e)
        return 0

    def chmod(self, path, mode):
        if hasattr(self.fs, "chmod"):
            path = "".join([self.root, path.lstrip("/")])
            return self.fs.chmod(path, mode)
        raise NotImplementedError


def run(
    fs,
    path,
    mount_point,
    foreground=True,
    threads=False,
    ready_file=False,
    ops_class=FUSEr,
):
    """Mount stuff in a local directory

    This uses fusepy to make it appear as if a given path on an fsspec
    instance is in fact resident within the local file-system.

    This requires that fusepy by installed, and that FUSE be available on
    the system (typically requiring a package to be installed with
    apt, yum, brew, etc.).

    Parameters
    ----------
    fs: file-system instance
        From one of the compatible implementations
    path: str
        Location on that file-system to regard as the root directory to
        mount. Note that you typically should include the terminating "/"
        character.
    mount_point: str
        An empty directory on the local file-system where the contents of
        the remote path will appear.
    foreground: bool
        Whether or not calling this function will block. Operation will
        typically be more stable if True.
    threads: bool
        Whether or not to create threads when responding to file operations
        within the mounter directory. Operation will typically be more
        stable if False.
    ready_file: bool
        Whether the FUSE process is ready. The ``.fuse_ready`` file will
        exist in the ``mount_point`` directory if True. Debugging purpose.
    ops_class: FUSEr or Subclass of FUSEr
        To override the default behavior of FUSEr. For Example, logging
        to file.

    """
    func = lambda: FUSE(
        ops_class(fs, path, ready_file=ready_file),
        mount_point,
        nothreads=not threads,
        foreground=foreground,
    )
    if not foreground:
        th = threading.Thread(target=func)
        th.daemon = True
        th.start()
        return th
    else:  # pragma: no cover
        try:
            func()
        except KeyboardInterrupt:
            pass


def main(args):
    """Mount filesystem from chained URL to MOUNT_POINT.

    Examples:

    python3 -m fsspec.fuse memory /usr/share /tmp/mem

    python3 -m fsspec.fuse local /tmp/source /tmp/local \\
            -l /tmp/fsspecfuse.log

    You can also mount chained-URLs and use special settings:

    python3 -m fsspec.fuse 'filecache::zip::file://data.zip' \\
            / /tmp/zip \\
            -o 'filecache-cache_storage=/tmp/simplecache'

    You can specify the type of the setting by using `[int]` or `[bool]`,
    (`true`, `yes`, `1` represents the Boolean value `True`):

    python3 -m fsspec.fuse 'simplecache::ftp://ftp1.at.proftpd.org' \\
            /historic/packages/RPMS /tmp/ftp \\
            -o 'simplecache-cache_storage=/tmp/simplecache' \\
            -o 'simplecache-check_files=false[bool]' \\
            -o 'ftp-listings_expiry_time=60[int]' \\
            -o 'ftp-username=anonymous' \\
            -o 'ftp-password=xieyanbo'
    """

    class RawDescriptionArgumentParser(argparse.ArgumentParser):
        def format_help(self):
            usage = super().format_help()
            parts = usage.split("\n\n")
            parts[1] = self.description.rstrip()
            return "\n\n".join(parts)

    parser = RawDescriptionArgumentParser(prog="fsspec.fuse", description=main.__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("url", type=str, help="fs url")
    parser.add_argument("source_path", type=str, help="source directory in fs")
    parser.add_argument("mount_point", type=str, help="local directory")
    parser.add_argument(
        "-o",
        "--option",
        action="append",
        help="Any options of protocol included in the chained URL",
    )
    parser.add_argument(
        "-l", "--log-file", type=str, help="Logging FUSE debug info (Default: '')"
    )
    parser.add_argument(
        "-f",
        "--foreground",
        action="store_false",
        help="Running in foreground or not (Default: False)",
    )
    parser.add_argument(
        "-t",
        "--threads",
        action="store_false",
        help="Running with threads support (Default: False)",
    )
    parser.add_argument(
        "-r",
        "--ready-file",
        action="store_false",
        help="The `.fuse_ready` file will exist after FUSE is ready. "
        "(Debugging purpose, Default: False)",
    )
    args = parser.parse_args(args)

    kwargs = {}
    for item in args.option or []:
        key, sep, value = item.partition("=")
        if not sep:
            parser.error(message=f"Wrong option: {item!r}")
        val = value.lower()
        if val.endswith("[int]"):
            value = int(value[: -len("[int]")])
        elif val.endswith("[bool]"):
            value = val[: -len("[bool]")] in ["1", "yes", "true"]

        if "-" in key:
            fs_name, setting_name = key.split("-", 1)
            if fs_name in kwargs:
                kwargs[fs_name][setting_name] = value
            else:
                kwargs[fs_name] = {setting_name: value}
        else:
            kwargs[key] = value

    if args.log_file:
        logging.basicConfig(
            level=logging.DEBUG,
            filename=args.log_file,
            format="%(asctime)s %(message)s",
        )

        class LoggingFUSEr(FUSEr, LoggingMixIn):
            pass

        fuser = LoggingFUSEr
    else:
        fuser = FUSEr

    fs, url_path = url_to_fs(args.url, **kwargs)
    logger.debug("Mounting %s to %s", url_path, str(args.mount_point))
    run(
        fs,
        args.source_path,
        args.mount_point,
        foreground=args.foreground,
        threads=args.threads,
        ready_file=args.ready_file,
        ops_class=fuser,
    )


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
