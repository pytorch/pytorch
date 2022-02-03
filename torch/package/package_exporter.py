from ._zip_file_torchscript import TorchScriptPackageZipFileWriter
from .package_exporter_oss import PackageExporter as PE
from .package_exporter_oss import * # noqa: F403

class PackageExporter(PE):
    """
    Shim for torch.package.PackageExporter in order to maintain BC.
    """
    importer: Importer

    def __init__(
        self,
        f: Union[str, Path, BinaryIO],
        importer: Union[Importer, Sequence[Importer]] = sys_importer
    ):
        """
        Create an exporter.

        Args:
            f: The location to export to. Can be a  ``string``/``Path`` object containing a filename
                or a binary I/O object.
            importer: If a single Importer is passed, use that to search for modules.
                If a sequence of importers are passsed, an ``OrderedImporter`` will be constructed out of them.
        """
        super.__init__(f, importer, TorchScriptPackageZipFileWriter)
