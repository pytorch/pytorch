# Testing

Use our test class and test runner:

```
from torch.testing._internal.common_utils import run_tests, TestCase

class TestFeature(TestCase):
    ...

if __name__ == "__main__":
    run_tests()
```

To test Tensor equality, use assertEqual.

# Commit messages

Don't commit unless the user explicitly asks you to.

When writing a commit message, don't make a bullet list of the individual
changes. Instead, if the PR is large, explain the order to review changes
(e.g., the logical progression), or if it's short just omit the bullet list
entirely.

Disclose that the PR was authored with Claude.

# Dynamo Config

Use `torch._dynamo.config.patch` context manager instead of manual try/finally for temporarily changing config:

```python
# Good - use patch context manager
with torch._dynamo.config.patch(force_compile_during_fx_trace=True):
    # test code here
    pass

# Bad - manual save/restore
orig = torch._dynamo.config.force_compile_during_fx_trace
try:
    torch._dynamo.config.force_compile_during_fx_trace = True
    # test code here
finally:
    torch._dynamo.config.force_compile_during_fx_trace = orig
```
