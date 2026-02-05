# Backward Compatibility Guidelines

This document covers backward compatibility (BC) considerations for PyTorch PR reviews.

## What Constitutes a BC-Breaking Change

### API Changes

| Change Type | BC Impact | Action Required |
|-------------|-----------|-----------------|
| Removing a public function/class | Breaking | Deprecation period required |
| Renaming a public API | Breaking | Deprecation period required |
| Changing function signature (removing/reordering args) | Breaking | Deprecation period required |
| Adding required arguments without defaults | Breaking | Add default value instead |
| Changing argument defaults | Potentially breaking | Document in release notes |
| Changing return type | Breaking | Deprecation period required |
| Removing, renaming or updating private API | Potentially Breaking | Validate no usage outside of PyTorch Core via global github search |

### Behavioral Changes

| Change Type | BC Impact | Action Required |
|-------------|-----------|-----------------|
| Raising new exceptions | Potentially breaking | Validate it is expected and document |
| Changing exception types | Potentially breaking | Document in release notes |
| Changing default device | Breaking | Explicit migration |
| Any user-visible change of existing behavior | Potentially breaking | Should be classified bug-fix or bc-breaking |

### What Is a Public API

Per the [official PyTorch Public API definition](https://github.com/pytorch/pytorch/wiki/Public-API-definition-and-documentation):

An API is **public** if:
- It's name does not start with an `_`
- Its submodule as reported by `__module__` starts with `"torch."`
- Its submodule where no name in the path starts with underscore

**Key rule**: If a function "looks public" and is documented on pytorch.org/docs, it is public. Undocumented functions that appear public may be changed or removed without deprecation.

## Python Version Support

PyTorch supports all non-EOL CPython versions which means the last 5 versions.


## When BC Breaks Are Acceptable

### With Proper Deprecation

BC-breaking changes are acceptable when:

1. **Deprecation warning added** - At least one release with deprecation warning
2. **Migration path documented** - Users know how to update their code
3. **Release notes updated** - Change is clearly documented
4. **Justified benefit** - The breaking change provides significant improvement

### Deprecation Pattern

```python
import warnings

def old_function(x, old_arg=None, new_arg=None):
    if old_arg is not None:
        warnings.warn(
            "old_arg is deprecated and will be removed in a future release. "
            "Use new_arg instead.",
            FutureWarning,
            stacklevel=2,
        )
        new_arg = old_arg
    # ... rest of implementation
```

### Without Deprecation (Rare)

Immediate BC breaks may be acceptable for:

- Security vulnerabilities
- Serious bugs that make the API unusable
- APIs explicitly marked experimental/beta
- Changes during a major version bump (e.g., 2.x to 3.0)

## Common BC Pitfalls

### 1. Changing Function Signatures

**Bad:**
```python
# Before
def forward(self, x, y):
    ...

# After - breaks callers using positional args
def forward(self, x, z, y):
    ...
```

**Good:**
```python
# After - add new args at end with defaults
def forward(self, x, y, z=None):
    ...
```

### 2. Removing Public Attributes

**Bad:**
```python
# Removing an attribute users might access
class Module:
    # self.weight removed
    pass
```

**Good:**
```python
class Module:
    @property
    def weight(self):
        warnings.warn("weight is deprecated", FutureWarning)
        return self._new_weight_implementation
```

### 3. Changing Default Behavior

**Bad:**
```python
# Silently changing default from False to True
def function(x, normalize=True):  # Was normalize=False
    ...
```

**Good:**
```python
def function(x, normalize=None):
    if normalize is None:
        warnings.warn(
            "normalize default is changing from False to True in v2.5",
            FutureWarning,
        )
        normalize = False  # Keep old default during deprecation
    ...
```

### 4. Changing Exception Types

**Bad:**
```python
# Users catching ValueError will miss the new exception
raise TypeError("...")  # Was ValueError
```

**Good:**
```python
# Create exception hierarchy or keep compatible
class NewError(ValueError):  # Inherits from old type
    pass
raise NewError("...")
```

### 5. Changing Output Shape or Dtype

**Bad:**
```python
# Silently returning different shape
return x.squeeze()  # Was returning x.unsqueeze(0)
```

**Good:**
```python
# Add explicit parameter for new behavior
def function(x, keepdim=None):
    if keepdim is None:
        warnings.warn("keepdim default changing to True", FutureWarning)
        keepdim = False
    ...
```

## Review Checklist for BC

When reviewing a PR, check:

- [ ] **No removed public APIs** - Or proper deprecation path exists
- [ ] **No changed signatures** - Or new args have defaults
- [ ] **No changed defaults** - Or deprecation warning added
- [ ] **No changed return types/shapes** - Or migration path documented
- [ ] **No changed exception types** - Or new types inherit from old
- [ ] **Deprecation uses FutureWarning** - Not DeprecationWarning (for user-facing APIs)
- [ ] **Deprecation has stacklevel=2** - Points to user code, not library internals

## Questions to Ask

When unsure about BC impact:

1. Would existing user code break silently (worst case)?
2. Would existing user code raise an exception (recoverable)?
3. Is there a migration path that doesn't require users to change code immediately?
4. Is this change documented in release notes?
