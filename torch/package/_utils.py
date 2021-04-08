from .importer import Importer, sys_importer

from ._mangling import is_mangled

def _import_module(module_name: str, importer: Importer=sys_importer):
    try:
        return importer.import_module(module_name)
    except ModuleNotFoundError as e:
        if not is_mangled(module_name):
            raise
        msg = (
            f"Module not found: '{module_name}'. Modules imported "
            "from a torch.package cannot be re-exported directly."
        )
        raise ModuleNotFoundError(msg) from None
