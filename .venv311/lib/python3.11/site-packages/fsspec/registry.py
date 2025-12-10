from __future__ import annotations

import importlib
import types
import warnings

__all__ = ["registry", "get_filesystem_class", "default"]

# internal, mutable
_registry: dict[str, type] = {}

# external, immutable
registry = types.MappingProxyType(_registry)
default = "file"


def register_implementation(name, cls, clobber=False, errtxt=None):
    """Add implementation class to the registry

    Parameters
    ----------
    name: str
        Protocol name to associate with the class
    cls: class or str
        if a class: fsspec-compliant implementation class (normally inherits from
        ``fsspec.AbstractFileSystem``, gets added straight to the registry. If a
        str, the full path to an implementation class like package.module.class,
        which gets added to known_implementations,
        so the import is deferred until the filesystem is actually used.
    clobber: bool (optional)
        Whether to overwrite a protocol with the same name; if False, will raise
        instead.
    errtxt: str (optional)
        If given, then a failure to import the given class will result in this
        text being given.
    """
    if isinstance(cls, str):
        if name in known_implementations and clobber is False:
            if cls != known_implementations[name]["class"]:
                raise ValueError(
                    f"Name ({name}) already in the known_implementations and clobber "
                    f"is False"
                )
        else:
            known_implementations[name] = {
                "class": cls,
                "err": errtxt or f"{cls} import failed for protocol {name}",
            }

    else:
        if name in registry and clobber is False:
            if _registry[name] is not cls:
                raise ValueError(
                    f"Name ({name}) already in the registry and clobber is False"
                )
        else:
            _registry[name] = cls


# protocols mapped to the class which implements them. This dict can be
# updated with register_implementation
known_implementations = {
    "abfs": {
        "class": "adlfs.AzureBlobFileSystem",
        "err": "Install adlfs to access Azure Datalake Gen2 and Azure Blob Storage",
    },
    "adl": {
        "class": "adlfs.AzureDatalakeFileSystem",
        "err": "Install adlfs to access Azure Datalake Gen1",
    },
    "arrow_hdfs": {
        "class": "fsspec.implementations.arrow.HadoopFileSystem",
        "err": "pyarrow and local java libraries required for HDFS",
    },
    "async_wrapper": {
        "class": "fsspec.implementations.asyn_wrapper.AsyncFileSystemWrapper",
    },
    "asynclocal": {
        "class": "morefs.asyn_local.AsyncLocalFileSystem",
        "err": "Install 'morefs[asynclocalfs]' to use AsyncLocalFileSystem",
    },
    "asyncwrapper": {
        "class": "fsspec.implementations.asyn_wrapper.AsyncFileSystemWrapper",
    },
    "az": {
        "class": "adlfs.AzureBlobFileSystem",
        "err": "Install adlfs to access Azure Datalake Gen2 and Azure Blob Storage",
    },
    "blockcache": {"class": "fsspec.implementations.cached.CachingFileSystem"},
    "box": {
        "class": "boxfs.BoxFileSystem",
        "err": "Please install boxfs to access BoxFileSystem",
    },
    "cached": {"class": "fsspec.implementations.cached.CachingFileSystem"},
    "dask": {
        "class": "fsspec.implementations.dask.DaskWorkerFileSystem",
        "err": "Install dask distributed to access worker file system",
    },
    "data": {"class": "fsspec.implementations.data.DataFileSystem"},
    "dbfs": {
        "class": "fsspec.implementations.dbfs.DatabricksFileSystem",
        "err": "Install the requests package to use the DatabricksFileSystem",
    },
    "dir": {"class": "fsspec.implementations.dirfs.DirFileSystem"},
    "dropbox": {
        "class": "dropboxdrivefs.DropboxDriveFileSystem",
        "err": (
            'DropboxFileSystem requires "dropboxdrivefs","requests" and "'
            '"dropbox" to be installed'
        ),
    },
    "dvc": {
        "class": "dvc.api.DVCFileSystem",
        "err": "Install dvc to access DVCFileSystem",
    },
    "file": {"class": "fsspec.implementations.local.LocalFileSystem"},
    "filecache": {"class": "fsspec.implementations.cached.WholeFileCacheFileSystem"},
    "ftp": {"class": "fsspec.implementations.ftp.FTPFileSystem"},
    "gcs": {
        "class": "gcsfs.GCSFileSystem",
        "err": "Please install gcsfs to access Google Storage",
    },
    "gdrive": {
        "class": "gdrive_fsspec.GoogleDriveFileSystem",
        "err": "Please install gdrive_fs for access to Google Drive",
    },
    "generic": {"class": "fsspec.generic.GenericFileSystem"},
    "gist": {
        "class": "fsspec.implementations.gist.GistFileSystem",
        "err": "Install the requests package to use the gist FS",
    },
    "git": {
        "class": "fsspec.implementations.git.GitFileSystem",
        "err": "Install pygit2 to browse local git repos",
    },
    "github": {
        "class": "fsspec.implementations.github.GithubFileSystem",
        "err": "Install the requests package to use the github FS",
    },
    "gs": {
        "class": "gcsfs.GCSFileSystem",
        "err": "Please install gcsfs to access Google Storage",
    },
    "hdfs": {
        "class": "fsspec.implementations.arrow.HadoopFileSystem",
        "err": "pyarrow and local java libraries required for HDFS",
    },
    "hf": {
        "class": "huggingface_hub.HfFileSystem",
        "err": "Install huggingface_hub to access HfFileSystem",
    },
    "http": {
        "class": "fsspec.implementations.http.HTTPFileSystem",
        "err": 'HTTPFileSystem requires "requests" and "aiohttp" to be installed',
    },
    "https": {
        "class": "fsspec.implementations.http.HTTPFileSystem",
        "err": 'HTTPFileSystem requires "requests" and "aiohttp" to be installed',
    },
    "jlab": {
        "class": "fsspec.implementations.jupyter.JupyterFileSystem",
        "err": "Jupyter FS requires requests to be installed",
    },
    "jupyter": {
        "class": "fsspec.implementations.jupyter.JupyterFileSystem",
        "err": "Jupyter FS requires requests to be installed",
    },
    "lakefs": {
        "class": "lakefs_spec.LakeFSFileSystem",
        "err": "Please install lakefs-spec to access LakeFSFileSystem",
    },
    "libarchive": {
        "class": "fsspec.implementations.libarchive.LibArchiveFileSystem",
        "err": "LibArchive requires to be installed",
    },
    "local": {"class": "fsspec.implementations.local.LocalFileSystem"},
    "memory": {"class": "fsspec.implementations.memory.MemoryFileSystem"},
    "oci": {
        "class": "ocifs.OCIFileSystem",
        "err": "Install ocifs to access OCI Object Storage",
    },
    "ocilake": {
        "class": "ocifs.OCIFileSystem",
        "err": "Install ocifs to access OCI Data Lake",
    },
    "oss": {
        "class": "ossfs.OSSFileSystem",
        "err": "Install ossfs to access Alibaba Object Storage System",
    },
    "pyscript": {
        "class": "pyscript_fsspec_client.client.PyscriptFileSystem",
        "err": "Install requests (cpython) or run in pyscript",
    },
    "reference": {"class": "fsspec.implementations.reference.ReferenceFileSystem"},
    "root": {
        "class": "fsspec_xrootd.XRootDFileSystem",
        "err": (
            "Install fsspec-xrootd to access xrootd storage system. "
            "Note: 'root' is the protocol name for xrootd storage systems, "
            "not referring to root directories"
        ),
    },
    "s3": {"class": "s3fs.S3FileSystem", "err": "Install s3fs to access S3"},
    "s3a": {"class": "s3fs.S3FileSystem", "err": "Install s3fs to access S3"},
    "sftp": {
        "class": "fsspec.implementations.sftp.SFTPFileSystem",
        "err": 'SFTPFileSystem requires "paramiko" to be installed',
    },
    "simplecache": {"class": "fsspec.implementations.cached.SimpleCacheFileSystem"},
    "smb": {
        "class": "fsspec.implementations.smb.SMBFileSystem",
        "err": 'SMB requires "smbprotocol" or "smbprotocol[kerberos]" installed',
    },
    "ssh": {
        "class": "fsspec.implementations.sftp.SFTPFileSystem",
        "err": 'SFTPFileSystem requires "paramiko" to be installed',
    },
    "tar": {"class": "fsspec.implementations.tar.TarFileSystem"},
    "tos": {
        "class": "tosfs.TosFileSystem",
        "err": "Install tosfs to access ByteDance volcano engine Tinder Object Storage",
    },
    "tosfs": {
        "class": "tosfs.TosFileSystem",
        "err": "Install tosfs to access ByteDance volcano engine Tinder Object Storage",
    },
    "wandb": {"class": "wandbfs.WandbFS", "err": "Install wandbfs to access wandb"},
    "webdav": {
        "class": "webdav4.fsspec.WebdavFileSystem",
        "err": "Install webdav4 to access WebDAV",
    },
    "webhdfs": {
        "class": "fsspec.implementations.webhdfs.WebHDFS",
        "err": 'webHDFS access requires "requests" to be installed',
    },
    "zip": {"class": "fsspec.implementations.zip.ZipFileSystem"},
}

assert list(known_implementations) == sorted(known_implementations), (
    "Not in alphabetical order"
)


def get_filesystem_class(protocol):
    """Fetch named protocol implementation from the registry

    The dict ``known_implementations`` maps protocol names to the locations
    of classes implementing the corresponding file-system. When used for the
    first time, appropriate imports will happen and the class will be placed in
    the registry. All subsequent calls will fetch directly from the registry.

    Some protocol implementations require additional dependencies, and so the
    import may fail. In this case, the string in the "err" field of the
    ``known_implementations`` will be given as the error message.
    """
    if not protocol:
        protocol = default

    if protocol not in registry:
        if protocol not in known_implementations:
            raise ValueError(f"Protocol not known: {protocol}")
        bit = known_implementations[protocol]
        try:
            register_implementation(protocol, _import_class(bit["class"]))
        except ImportError as e:
            raise ImportError(bit.get("err")) from e
    cls = registry[protocol]
    if getattr(cls, "protocol", None) in ("abstract", None):
        cls.protocol = protocol

    return cls


s3_msg = """Your installed version of s3fs is very old and known to cause
severe performance issues, see also https://github.com/dask/dask/issues/10276

To fix, you should specify a lower version bound on s3fs, or
update the current installation.
"""


def _import_class(fqp: str):
    """Take a fully-qualified path and return the imported class or identifier.

    ``fqp`` is of the form "package.module.klass" or
    "package.module:subobject.klass".

    Warnings
    --------
    This can import arbitrary modules. Make sure you haven't installed any modules
    that may execute malicious code at import time.
    """
    if ":" in fqp:
        mod, name = fqp.rsplit(":", 1)
    else:
        mod, name = fqp.rsplit(".", 1)

    is_s3 = mod == "s3fs"
    mod = importlib.import_module(mod)
    if is_s3 and mod.__version__.split(".") < ["0", "5"]:
        warnings.warn(s3_msg)
    for part in name.split("."):
        mod = getattr(mod, part)

    if not isinstance(mod, type):
        raise TypeError(f"{fqp} is not a class")

    return mod


def filesystem(protocol, **storage_options):
    """Instantiate filesystems for given protocol and arguments

    ``storage_options`` are specific to the protocol being chosen, and are
    passed directly to the class.
    """
    if protocol == "arrow_hdfs":
        warnings.warn(
            "The 'arrow_hdfs' protocol has been deprecated and will be "
            "removed in the future. Specify it as 'hdfs'.",
            DeprecationWarning,
        )

    cls = get_filesystem_class(protocol)
    return cls(**storage_options)


def available_protocols():
    """Return a list of the implemented protocols.

    Note that any given protocol may require extra packages to be importable.
    """
    return list(known_implementations)
