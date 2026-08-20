import sys

from torch.utils._config_module import alias_fields_from, Config, install_config_module


# Fixtures for alias_fields_from exercising: field-type resolution (plain vs
# Config, annotated vs not), child-override independence, and mutable-default
# sharing. Installed (unlike a standalone class) so aliases actually resolve.
#
# Kept in a module of its own rather than fake_config_module.py: test_fuzzer.py
# sweeps every field of fake_config_module through ConfigFuzzer, which reads
# _ConfigEntry.default directly without resolving aliases (it is _UNSET_SENTINEL
# for every alias entry). That is harmless for bool-typed aliases but crashes
# sampling a list-typed one, so the mutable-default fixture below lives here
# instead.
class alias_parent:
    e_bool = True
    e_config_int: int = Config(default=7)
    e_config_no_annotation = Config(default=42)
    e_annotated: str = "hi"
    e_annotated_config: str | None = Config(default=None)
    e_list = [1, 2]

    def method_not_a_field(self):
        return None


# Every field of `alias_parent` not overridden here is aliased to it.
@alias_fields_from(alias_parent)
class alias_child:
    pass


# `e_bool` is this child's own field, independent of the parent; every other
# field is aliased.
@alias_fields_from(alias_parent)
class alias_child_override:
    e_bool = False


# Hand-written 2-hop chain: e_chained_alias -> fake_config_module.e_aliased_bool
# -> fake_config_module2.e_aliasing_bool. alias_fields_from rejects building a
# chain itself, but nothing stops a hand-written Config(alias=...) from
# pointing at an already-aliased field, so this exercises that path directly.
e_chained_alias: bool = Config(
    alias="torch.testing._internal.fake_config_module.e_aliased_bool"
)


install_config_module(sys.modules[__name__])
