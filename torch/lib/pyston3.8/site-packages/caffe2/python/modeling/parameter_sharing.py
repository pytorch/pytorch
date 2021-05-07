




from caffe2.python import scope

import contextlib
import logging

logger = logging.getLogger(__name__)


class ParameterSharingContext(object):
    """
    This class manages scope driven way of parameter sharing across different
    NameScopes.
    """

    def __init__(self):
        self._scope_overrides = {}
        self._contexts = []

    def _resolve_scope_overrides(self, candidate_scope):
        """
        Recursively resolves all scope overrides, i.e multiple steps of
        override can be used.

        For example, if one provides following scope overrides:
        {'scope_b': 'scope_a'} and within 'scope_b' - {'shared_child': ''},
        then name 'w' will get resolved to the following blobs depending on the
        namescope:
          a. 'scope_a' -> 'scope_a/w'
          b. 'scope_b' -> 'scope_a/w'
          c. 'scope_c' -> 'scope_c/w'
          d. 'scope_b/shared_child' -> 'scope_a/w'
          d. 'scope_b/unshared_child' -> 'scope_a/unshared_child/w'
        """
        best_scope = candidate_scope
        best_scope_idx = 0
        sub_scopes = candidate_scope.split(scope._NAMESCOPE_SEPARATOR)

        cur_scope = ''
        for idx, sub_scope in enumerate(sub_scopes):
            cur_scope = cur_scope + sub_scope + scope._NAMESCOPE_SEPARATOR
            if cur_scope in self._scope_overrides:
                best_scope = self._scope_overrides[cur_scope]
                best_scope_idx = idx
        if best_scope == candidate_scope:
            return candidate_scope
        else:
            return (self._resolve_scope_overrides(best_scope) +
                    scope._NAMESCOPE_SEPARATOR.join(
                        sub_scopes[best_scope_idx + 1:]))

    def get_parameter_name(self, name):
        candidate_scope = scope.CurrentNameScope()
        best_scope = self._resolve_scope_overrides(candidate_scope)
        if best_scope != candidate_scope:
            logger.info("Overwriting scope {0} with scope {1}".format(
                candidate_scope, best_scope))

        return best_scope + name

    def add_scope_overrides(self, shared_scopes):
        self._contexts.append(shared_scopes)
        self._scope_overrides.update(shared_scopes)

    def pop(self):
        assert len(self._contexts) > 0
        self._contexts.pop()
        self._scope_overrides = {}
        for x in self._contexts:
            self._scope_overrides.update(x)


parameter_sharing_context = ParameterSharingContext()


def _normalize_namescope(namescope):
    if namescope and namescope[-1] != scope._NAMESCOPE_SEPARATOR:
        return namescope + scope._NAMESCOPE_SEPARATOR
    else:
        return namescope


@contextlib.contextmanager
def ParameterSharing(shared_scopes):
    """
    Helper function for sharing scopes.
    All the parameters within the shared_scopes, will be remapped with the
    respect of CurrentNamescope()

    I.e. if one calls ParameterSharing with {'scope_b': 'scope_'a'}, from the
    scope 'some_global_scope', it'll effectively mean, that all parameters from
    'some_global_scope/scope_b' will shared with the parameters from
    'some_global_scope/scope_a'
    """
    assert isinstance(shared_scopes, dict)

    shared_scope_overrides = {}
    current_scope = scope.CurrentNameScope()
    for k, v in shared_scopes.items():
        assert not v.startswith(k), (
            "Illegal override for parameter sharing. {} is prefix of {}".
            format(k, v))
        k = current_scope + k
        v = current_scope + v
        # Normalize all the scopes, so scope_a and scope_a/ are equivalent
        k = _normalize_namescope(k)
        v = _normalize_namescope(v)
        shared_scope_overrides[k] = v

    try:
        parameter_sharing_context.add_scope_overrides(shared_scope_overrides)
        yield
    finally:
        parameter_sharing_context.pop()
