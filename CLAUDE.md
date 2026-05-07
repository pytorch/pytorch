# Scratch Space

Use `agent_space/` (git-ignored, at repo root) for temporary scripts, scratch files, and throwaway experiments. Do not commit files from this directory.

# PR Review

When asked to review a PR, always use the /pr-review skill.

# Environment

If any tool you're trying to use (pip, python, spin, etc) is missing, check for
a `.venv` directory in the project root or its parent directory. If found,
activate it and retry. If no `.venv` is found, stop and ask the user if an
environment is needed. Do NOT try to find alternatives or install these tools.

# CI Docker Images

The `.ci/docker/` directory is content-hashed to determine whether Docker images
need rebuilding. Any file change inside `.ci/docker/` (including the README)
changes the hash and triggers a full Docker image rebuild. Do not make changes
in this directory unless you intend to rebuild Docker images. When Docker builds
are broken (e.g., due to an upstream Ubuntu outage), avoid touching this
directory so you don't force a rebuild against the broken state.

# Build

Always check local memory for build configuration (env vars, incremental-build shortcuts, etc.) before running the build, and apply what you find. If nothing applicable is in memory, ask the user.
All build (both codegen, C++ and python) is done via `pip install -e . -v --no-build-isolation`.
You should NEVER run any other command to build PyTorch.

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
For tests over multiple inputs, use the `@parametrize` decorator.
For any test that checks numerics of the on-device implementation, use `instantiate_device_type_tests` to write device-generic tests.

# Linting

Only use commands provided via `spin` for linting.
Use `spin help` to list available commands.
Generally, use `spin lint` as to run the lint and `spin fixlint` to apply automatic fixes.

When the user asks you to commit or amend, run `lintrunner -a` before creating
the commit. Fix any lint errors it reports, then commit.

# Commit messages

Don't commit unless the user explicitly asks you to.

When writing a commit message, don't make a bullet list of the individual
changes. Instead, if the PR is large, explain the order to review changes
(e.g., the logical progression), or if it's short just omit the bullet list
entirely.

Disclose that the PR was authored with an AI assistant. Do this informally in
the commit body (e.g., "Authored by Claude." or a similar attribution for
whichever assistant was used). NEVER add a `Co-authored-by:` trailer
attributing the AI assistant, as it interferes with the Linux Foundation CLA
bot.

When the user asks you to amend a commit, check whether the commit message
still accurately describes the changes. If it doesn't and the commit is not a
ghstack commit, update the message. For ghstack commits, amending the message
is a no-op, so just remind the user to update the PR description if needed.

If a commit message contains `ghstack-source-id` or `Pull-Request` trailers,
you MUST preserve them when rewriting or splitting commit messages. ghstack
will update the source id automatically when needed.

# ghstack Workflow

ghstack commits follow a different workflow than the conventional GitHub branch
and PR workflow. First identify whether you're on a ghstack commit:

- If HEAD is a detached commit, you are almost certainly in a ghstack flow.
- If the commit message contains a `ghstack-source-id` trailer, it is an
  existing ghstack commit.
- If the commit is associated with a remote branch like `origin/gh/USERNAME/N`,
  it is likely a ghstack commit (imperfect signal: local amends without a push
  can desync this).

Rules for working with ghstack:

- **Don't amend unless asked.** If the user asks you to work on a ghstack
  commit, leave changes uncommitted so the user can review with `git diff`.
  Only amend into the commit if the user explicitly asks you to amend or to
  submit it directly.
- **Submitting.** Run `ghstack` to submit. When only working on a single
  commit, use `ghstack --no-stack` to avoid updating the rest of the stack and
  burning unnecessary CI. Use a full `ghstack` when you're intentionally
  updating CI for the whole stack.
- **Preserve metadata trailers.** When editing a commit message, never delete
  `Pull-Request:` or `ghstack-source-id:` trailers. If you modified the commit
  message, run `ghstack -u` afterwards to push the updated PR description.
- **Never push directly.** Do not `git push` to branches, and never directly
  modify the `gh/USERNAME/N` branches — ghstack manages those.
- **Finding the PR.** If the user asks to pull CI results or code review for a
  ghstack commit, get the PR URL from the `Pull-Request` trailer in the commit
  message. Use `gh` CLI to fetch status/comments from there.
- **Editing earlier commits / splitting.** Treat it like a normal stack of
  commits (use `git rebase`, etc.). Commits that keep their metadata trailers
  stay associated with their existing PRs; commits without trailers will get a
  fresh PR on submit. A full `ghstack` run is usually appropriate here.

# Coding Style Guidelines

Follow these rules for all code changes in this repository:

- Minimize comments; be concise; code should be self-explanatory and self-documenting.
- Comments should be useful, for example, comments that remind the reader about
  some global context that is non-obvious and can't be inferred locally.
- Don't make trivial (1-2 LOC) helper functions that are only used once unless
  it significantly improves code readability.
- Prefer clear abstractions. State management should be explicit.
  For example, if managing state in a Python class: there should be a clear
  class definition that has all of the members: don't dynamically `setattr`
  a field on an object and then dynamically `getattr` the field on the object.
- Match existing code style and architectural patterns.
- Assume the reader has familiarity with PyTorch. They may not be the expert
  on the code that is being read, but they should have some experience in the
  area.
- ASCII only in newly added code comments. Do not introduce Unicode characters
  (e.g., smart quotes, em dashes, arrows, non-ASCII letters) in new comments.
  Leave preexisting Unicode in untouched comments alone; only enforce this for
  comments you are adding or rewriting.

If uncertain, choose the simpler, more concise implementation.

# Dynamo Config

Use `torch._dynamo.config.patch` for temporarily changing config. It can be used as a decorator on test methods or as a context manager:

```python
# Good - use patch as decorator on test method
@torch._dynamo.config.patch(force_compile_during_fx_trace=True)
def test_my_feature(self):
    # test code here
    pass

# Good - use patch as context manager
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

# Fixing B950 line too long in multi-line string blocks

If B950 line too long triggers on a multi-line string block, you cannot fix it by
putting # noqa: B950 on that line directly, as that would change the meaning of the
string, nor can you fix it by line breaking the string (since you need the string
to stay the same).  Instead, put # noqa: B950 on the same line as the terminating
triple quote.

Example:

```
    self.assertExpectedInline(
        foo(),
        """
this line is too long...
""",  # noqa: B950
    )
```

# Logging and Structured Tracing

When adding debug logging for errors or diagnostic info, consider two user personas:

1. **Local development**: Users run locally and can access files on disk
2. **Production jobs**: Users can only access logs via `tlparse` from structured traces

For production debugging, use `trace_structured` to log artifacts:

```python
from torch._logging import trace_structured

# Log an artifact (graph, edge list, etc.)
trace_structured(
    "artifact",
    metadata_fn=lambda: {
        "name": "my_debug_artifact",
        "encoding": "string",
    },
    payload_fn=lambda: my_content_string,
)
```

To check if structured tracing is enabled (for conditional messaging):

```python
from torch._logging._internal import trace_log

if trace_log.handlers:
    # Structured tracing is enabled, suggest tlparse in error messages
    msg += "[Use tlparse to extract debug artifacts]"
```

**Best practices for error diagnostics:**

- Always log to `trace_structured` for production (no runtime cost if disabled)
- If you're dumping debug info in the event of a true internal compiler exception,
  you can also consider writing to local files for local debugging convenience
- In error messages, tell users about both options:
  - Local files: "FX graph dump: min_cut_failed_graph.txt"
  - Production: "Use tlparse to extract artifacts" (only if tracing enabled)
- Use `_get_unique_path()` pattern to avoid overwriting existing debug files

# cuda::ptx

When using `<cuda/ptx>` typed wrappers for PTX instructions:

- **Namespace**: Inside `namespace at::native`, unqualified `cuda::ptx` resolves
  to the sibling `at::cuda` namespace. Always use `::cuda::ptx` or alias it:
  `namespace ptx = ::cuda::ptx;`
- **Include conflicts**: The monolithic `<cuda/ptx>` header can fail when included
  alongside heavy PyTorch headers (e.g. `Loops.cuh`) due to CCCL bugs in
  transitive headers like `cp_async_bulk_tensor.h`. Workaround: put kernels using
  `<cuda/ptx>` in a separate `.cu` file with minimal includes.
- **mbarrier_try_wait_parity is non-blocking**: `ptx::mbarrier_try_wait_parity()`
  returns `bool` (tries once). You must wrap it in a spin loop:
  `while (!ptx::mbarrier_try_wait_parity(mbar, parity)) {}`
- **Half/BFloat16 types**: `cuda::ptx` overloads use CUDA native types (`__half`,
  `__nv_bfloat16`), not PyTorch wrappers (`c10::Half`, `c10::BFloat16`).
  Use `reinterpret_cast` at the call site.
- **cp_async_bulk_wait_group**: Takes a compile-time constant via
  `ptx::n32_t<N>{}`, not a runtime integer.
- **Mbarrier smem**: Mbarrier memory must never alias with data targeted by TMA
  operations. Place mbarriers in a separate smem region from data buffers.
