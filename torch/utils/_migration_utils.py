# helper function for warnings for AO migration, see
# https://github.com/pytorch/pytorch/issues/81667 for context
def _get_ao_migration_warning_str(
    module__name__,
    object_name,
) -> str:
    old_name = f"{module__name__}.{object_name}"
    new_root = module__name__.replace("torch", "torch.ao")
    new_name = f"{new_root}.{object_name}"
    s = (
        f"`{old_name}` has been moved to `{new_name}`, and the "
        f"`{old_name}` syntax will be removed in a future version "
        "of PyTorch. Please see "
        "https://github.com/pytorch/pytorch/issues/81667 for detailed "
        "migration instructions."
    )
    return s

# Note: this does not start with an underscore because we need to
# keep statements such as `from torch.quantization.foo import *`
# backwards compatible, and a low-effort way to do this is to allow
# the deprecated objects to still participate in the `import *`
# syntax.
_AO_MIGRATION_DEPRECATED_NAME_PREFIX = "deprecated"
