import sys
import importlib
import importlib.abc
import importlib.machinery

_EINOPS_061_NEEDS_PATCH = True
_TARGET_VERSIONS = {"0.6.1", "0.6.2rc0", "0.7.0rc1"}


def _remove_patch_finder():
    for i, f in enumerate(sys.meta_path):
        if isinstance(f, _EinopsImportIntercept):
            sys.meta_path.pop(i)
            break


def _apply_patch():
    """
    Apply the einops patch to target versions only if einops is already loaded.
    Noop if einops isn't imported yet or already patched.
    Never try to apply patch again when einops loaded with different version than targets.
    """
    global _EINOPS_061_NEEDS_PATCH
    if not _EINOPS_061_NEEDS_PATCH:
        return

    einops_module = sys.modules.get("einops")

    if not einops_module:
        return

    try:
        ver = getattr(einops_module, "__version__", "")

        if ver not in _TARGET_VERSIONS:
            _EINOPS_061_NEEDS_PATCH = False
            _remove_patch_finder()
            return

        einops_module = sys.modules.get("einops.einops") or importlib.import_module(
            "einops.einops"
        )

        if hasattr(einops_module, "_reconstruct_from_shape_uncached") and hasattr(
            einops_module, "_reconstruct_from_shape"
        ):
            einops_module._reconstruct_from_shape = (
                einops_module._reconstruct_from_shape_uncached
            )

        _prepare_transformation_recipe = getattr(
            einops_module, "_prepare_transformation_recipe", None
        )
        if callable(_prepare_transformation_recipe) and hasattr(
            _prepare_transformation_recipe, "__wrapped__"
        ):
            einops_module._prepare_transformation_recipe = (
                _prepare_transformation_recipe.__wrapped__
            )

        _EINOPS_061_NEEDS_PATCH = False
        _remove_patch_finder()

    except Exception:
        _EINOPS_061_NEEDS_PATCH = False
        _remove_patch_finder()
        return


class _EinopsPatchLoader(importlib.abc.Loader):
    def __init__(self, wrapped_loader: importlib.abc.Loader):
        self._wrapped_loader = wrapped_loader

    def exec_module(self, module):
        self._wrapped_loader.exec_module(module)

        _apply_patch()


class _EinopsImportIntercept(importlib.abc.MetaPathFinder):
    """Intercept einops and submodules imports to apply the patch"""

    def find_spec(self, fullname, path, target=None):
        if fullname == "einops" or fullname.startswith("einops."):
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec and spec.loader and isinstance(spec.loader, importlib.abc.Loader):
                spec.loader = _EinopsPatchLoader(spec.loader)
                return spec
        return None


def _patch_einops_061():
    """
    torch.compile fix for einops 0.6.1, 0.6.2rc0 and 0.7.0rc1
    https://github.com/pytorch/pytorch/issues/157417

    - If einops is already imported, patch immediately
    - If not imported, register a meta_path finder to patch at first import

    TODO: verify if this patch can be removed once SymInt.__hash__ doesn't throw TypeError
    and returns a valid hash that satisfies or skips SymInt heap check
    https://github.com/pytorch/pytorch/blob/9e792f583afae92ec8ddedac1660bd79991d1f4f/c10/core/SymIntArrayRef.h#L41
    """

    _apply_patch()

    if _EINOPS_061_NEEDS_PATCH and not any(
        isinstance(f, _EinopsImportIntercept) for f in sys.meta_path
    ):
        sys.meta_path.insert(0, _EinopsImportIntercept())
