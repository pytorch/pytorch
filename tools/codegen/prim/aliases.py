from typing import Tuple, Optional

# Note [primTorch Build]
# Some primTorch source files are available at both build and runtime. This
#   allows them to be used as datasources when building and be available
#   for programmatic inspection after PyTorch is installed.
# To do this, PyTorch's build system copies some primTorch files
#   from tools/codegen/prim to torch/prim, where they override placeholders.
# Because of this, this file can only import Python packages and other
#   primTorch components that are also available at both build and run time.

# Note [primTorch Aliases]

# Describes an alias.
# Alias overloads will be constructed based on overloads of the operator specified
#   by alias_for. alias_for should be the base name (in native_functions.yaml) of
#   the aliased operator (AKA the "ATen name" of the operator).
# If namespace is specified then the alias will be created in that namespace
#   (e.g. torch.linalg). By default aliases are created in the torch namespace.
# If has_method is specified then every alias overload either will or won't
#   have a method variant (depending on the setting). See
#   the [Method variants] note in alias_gen.py for details on whether alias overloads
#   have method variants or not when has_method is unspecified.
# While primTorch handles native function and C++ generation, it does NOT handle every aspect
#   of adding an alias. In addition to adding the alias here, the following must be done:
#   1) The OpInfo entry for the aliased operator must exist, and at least one of its entries
#        must be updated to express the alias relationship. See test_alias_relationships in
#        test_ops.py.
#   2) Steps 4--6 of the [Adding an Alias] note must be followed, updating
#        the documentation, torch/overrides.py, and the alias_map in
#        torch/csrc/jit/passes/normalize_ops.cpp.

# TODO:
#   - don't require torch.overrides.py updates for ops with OpInfos
#   - automatically update the documentation to describe alias relationships
#   - automatically generate the alias_map in torch/csrc/jit/passes/normalize_ops.cpp

class AliasInfo(object):
    __slots__ = ('name', 'alias_for', 'namespace', 'has_method')

    # Arguments:
    #   - name, the ATen name of the alias. If in the torch namespace the name will be bound as torch.foo.
    #   - alias_for, the ATen "base name" of the aliased operator.
    #   - namespace, the namespace to add the operator to
    #   - has_method, allows overriding whether the alias should create method variants or not. See the
    #       det alias below for an example.
    def __init__(self, name: str, *, alias_for: str, namespace: str = None, has_method: Optional[bool] = None):
        self.name = name
        self.alias_for = alias_for
        self.namespace = namespace  # NOTE: unexercised
        self.has_method = has_method

# Sequence of all AliasInfos
alias_infos: Tuple[AliasInfo] = (
    AliasInfo('absolute', alias_for='abs'),
    AliasInfo('arccos', alias_for='acos'),
    AliasInfo('concat', alias_for='cat'),
    AliasInfo('det', alias_for='linalg_det', has_method=True),
    AliasInfo('clip', alias_for='clamp'),
    AliasInfo('moveaxis', alias_for='movedim'),
    AliasInfo('less', alias_for='lt'),
    AliasInfo('swapaxes', alias_for='transpose'),
)
