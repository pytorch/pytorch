# Intentionally fails to import. Used by test_utils_config_module.py to
# exercise the branch of _get_alias_module_and_name that must propagate a
# genuine import failure (e.g. a missing third-party dependency) instead of
# treating it as an unresolved alias. Raising directly (rather than a real
# `import some_missing_dep`) avoids tripping up tools that statically resolve
# imports.
raise ModuleNotFoundError(
    "No module named 'definitely_not_a_real_dependency'",
    name="definitely_not_a_real_dependency",
)
