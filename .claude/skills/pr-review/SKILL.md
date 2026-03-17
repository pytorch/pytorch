---
name: pr-review
description: Review PyTorch pull requests for code quality, test coverage, security, and backward compatibility. Use when reviewing PRs, when asked to review code changes, or when the user mentions "review PR", "code review", or "check this PR".
---

# PyTorch PR Review Skill

Review PyTorch pull requests focusing on what CI cannot check: code quality, test coverage adequacy, security vulnerabilities, and backward compatibility. Linting, formatting, type checking, and import ordering are handled by CI.

## Usage Modes

### No Argument

If the user invokes `/pr-review` with no arguments, **do not perform a review**. Instead, ask the user what they would like to review:

> What would you like me to review?
> - A PR number or URL (e.g., `/pr-review 12345`)
> - A local branch (e.g., `/pr-review branch`)

### Local CLI Mode

The user provides a PR number or URL:

```
/pr-review 12345
/pr-review https://github.com/pytorch/pytorch/pull/12345
```

For a detailed review with line-by-line specific comments:

```
/pr-review 12345 detailed
```

Use `gh` CLI to fetch PR data:

```bash
# Get PR details
gh pr view <PR_NUMBER> --json title,body,author,baseRefName,headRefName,files,additions,deletions,commits

# Get the diff
gh pr diff <PR_NUMBER>

# Get PR comments
gh pr view <PR_NUMBER> --json comments,reviews
```

### Local Branch Mode

Review changes in the current branch that are not in `main`:

```
/pr-review branch
/pr-review branch detailed
```

Use git commands to get branch changes:

```bash
# Get current branch name
git branch --show-current

# Get list of changed files compared to main
git diff --name-only main...HEAD

# Get full diff compared to main
git diff main...HEAD

# Get commit log for the branch
git log main..HEAD --oneline

# Get diff stats (files changed, insertions, deletions)
git diff --stat main...HEAD
```

For local branch reviews:
- The "Summary" should describe what the branch changes accomplish based on commit messages and diff
- Use the current branch name in the review header instead of a PR number
- All other review criteria apply the same as PR reviews

### GitHub Actions Mode

When invoked via `@claude /pr-review` on a GitHub PR, the action pre-fetches PR
metadata and injects it into the prompt. Detect this mode by the presence of
`<formatted_context>`, `<pr_or_issue_body>`, and `<comments>` tags in the prompt.

The prompt already contains:
- PR metadata (title, author, branch names, additions/deletions, file count)
- PR body/description
- All comments and review comments (with file/line references)
- List of changed files with paths and change types

Use git commands to get the diff and commit history. The base branch name is in the
prompt context (look for `PR Branch: <head> -> <base>` or the `baseBranch` field).

```bash
# Get the full diff against the base branch
git diff origin/<baseBranch>...HEAD

# Get diff stats
git diff --stat origin/<baseBranch>...HEAD

# Get commit history for this PR
git log origin/<baseBranch>..HEAD --oneline

# If the base branch ref is not available, fetch it first
git fetch origin <baseBranch> --depth=1
```

Do NOT use `gh` CLI commands in this mode -- only git commands are available.
All PR metadata, comments, and reviews are already in the prompt context;
only the diff and commit log need to be fetched via git.

## Review Philosophy

A single line of code can have deep cross-cutting implications: a missing device guard causes silent data corruption on multi-GPU, a missing `Composite` dispatch key breaks every out-of-tree backend, a manual dtype check instead of `TensorIterator` silently skips type promotion. **Treat every line as potentially load-bearing.**

Do not skim. Do not summarize the diff and move on. Read every changed line and ask: *does this interact with existing PyTorch infrastructure that the author may not know about?* When uncertain, **investigate** — spawn a sub-agent to read the surrounding code, the infrastructure the PR should be using, or the tests that should exist. The cost of a false negative (missing a real issue) is much higher than the cost of investigation.

## Review Workflow

### Step 1: Fetch PR Information

**Local CLI mode**: Use `gh` commands to get PR metadata, changed files, full diff,
existing comments/reviews, and associated issue information.

**Local Branch mode**: Use `git diff` and `git log` against `main` as shown in the
Local Branch Mode section above.

**GitHub Actions mode**: PR metadata, comments, and reviews are already in the prompt.
Use `git diff origin/<baseBranch>...HEAD` for the full diff and
`git log origin/<baseBranch>..HEAD --oneline` for the commit log.

### Step 2: Understand Context

Before reviewing, build understanding of what the PR touches and why:
1. Identify the purpose of the change from title/description/issue
2. Group changes by type (new code, tests, config, docs)
3. Note the scope of changes (files affected, lines changed)
4. **Spawn sub-agents to read the unchanged code surrounding each changed file.** The diff alone is not enough — you need to understand the existing patterns, base classes, and infrastructure in the files being modified. For each significantly changed file, a sub-agent should read the full file (or the relevant class/function) and report back: what patterns does this file follow? What infrastructure does it use? What invariants does it maintain?

### Step 3: Deep Review — Line-by-Line with Investigation

This is the core of the review. Go through **every changed line** in the diff and evaluate it against the review checklist in [review-checklist.md](review-checklist.md).

**How to use sub-agents during review:**

The checklist is large. You cannot hold the full context of every infrastructure system in your head. Instead, when you encounter a changed line that touches a checklist area, **spawn a sub-agent** to investigate whether the checklist item applies. For example:

- A PR adds a new C++ kernel → spawn a sub-agent to check: Does it use TensorIterator? DispatchStub? Structured kernels? AT_DISPATCH? Does it have a meta implementation? A Composite fallback?
- A PR adds a new test → spawn a sub-agent to check: Does an OpInfo exist for this op? Is the test device-generic? Does it use make_tensor, @dtypes, TestCase?
- A PR modifies autograd code → spawn a sub-agent to check: Is derivatives.yaml the right place? Does it use setup_context? Does it have gradcheck tests?
- A PR adds a new operator → spawn a sub-agent to check: Is it in native_functions.yaml? Does it have proper tags? A Composite dispatch? Meta/fake impls? Schema annotations?

**Spawn sub-agents in parallel** for independent investigation areas. A typical review of a medium PR should spawn 3-8 sub-agents. Large PRs touching multiple subsystems may need more.

**Checklist areas** (see [review-checklist.md](review-checklist.md) for full details):
- Code quality and design
- PyTorch infrastructure — C++ kernels (TensorIterator, DispatchStub, AT_DISPATCH, device guards), CUDA/device management, operator registration and codegen (native_functions.yaml, Composite dispatch, meta/fake implementations), autograd (derivatives.yaml, autograd.Function, gradcheck), Python utilities (pytree, __torch_function__, logging), nn module patterns, Dynamo/Inductor/compile, FX/export, type promotion, serialization, distributed, tensor subclasses
- Testing adequacy (OpInfo, ModuleInfo, device-generic tests, @dtypes, @parametrize, make_tensor)
- Security considerations
- Thread safety and concurrency (Python, C++, CPython C API, NoGIL)
- Performance implications
- Any behavior change not expected by author

### Step 4: Check Backward Compatibility

Evaluate BC implications. See [bc-guidelines.md](bc-guidelines.md) for:
- What constitutes a BC-breaking change
- Required deprecation patterns
- Common BC pitfalls

For non-trivial BC questions (e.g., "does changing this default break downstream users?"), spawn a sub-agent to search for existing callers of the modified API.

### Step 5: Formulate Review

Structure your review with actionable feedback organized by category. Every finding should be traceable to a specific line in the diff and a specific checklist item.

## Review Areas

| Area | Focus | Reference |
|------|-------|-----------|
| Code Quality | Abstractions, patterns, complexity | [review-checklist.md](review-checklist.md) |
| API Design | New patterns, flag-based access, broader implications | [review-checklist.md](review-checklist.md) |
| C++ Kernels | TensorIterator, DispatchStub, AT_DISPATCH, structured kernels, device guards, memory format | [review-checklist.md](review-checklist.md) |
| CUDA/Device | C10_CUDA_CHECK, stream/event guards, recordStream, CUDA graphs, AcceleratorHooks | [review-checklist.md](review-checklist.md) |
| Op Registration | native_functions.yaml, Composite fallback, meta/fake impls, tags, schema annotations | [review-checklist.md](review-checklist.md) |
| Autograd | derivatives.yaml, autograd.Function patterns, gradcheck, forward-mode AD, vmap | [review-checklist.md](review-checklist.md) |
| Python Utils | __torch_function__, pytree, logging, deprecation, backends context | [review-checklist.md](review-checklist.md) |
| nn Modules | ModuleList/Dict, nn.init, parametrize, state_dict versioning, LazyModule | [review-checklist.md](review-checklist.md) |
| Dynamo/Inductor | @register_lowering, decompositions, CustomGraphPass, config.patch, graph breaks | [review-checklist.md](review-checklist.md) |
| FX/Export | PassBase, PassManager, Interpreter, subgraph rewriter, ShapeProp, make_fx | [review-checklist.md](review-checklist.md) |
| Type Promotion | elementwise_dtypes, TensorIterator dtype handling, result_type, promoteTypes | [review-checklist.md](review-checklist.md) |
| Serialization | weights_only, safe_globals, skip_data | [review-checklist.md](review-checklist.md) |
| Distributed | DeviceMesh, distributed testing with MultiThreadedPG | [review-checklist.md](review-checklist.md) |
| Tensor Subclasses | _make_wrapper_subclass, __tensor_flatten__/__unflatten__ | [review-checklist.md](review-checklist.md) |
| Testing | OpInfo, ModuleInfo, device-generic, @dtypes, @parametrize, make_tensor | [review-checklist.md](review-checklist.md) |
| Security | Injection, credentials, input handling | [review-checklist.md](review-checklist.md) |
| Performance | Regressions, device handling, memory, profiling, benchmarking | [review-checklist.md](review-checklist.md) |
| Thread Safety | Data races, GIL assumptions, NoGIL, CPython C API | [review-checklist.md](review-checklist.md) |
| BC | Breaking changes, deprecation | [bc-guidelines.md](bc-guidelines.md) |

## Output Format

Structure your review as follows. **Omit sections where you have no findings** — don't write "No concerns" for every empty section. Only include sections with actual observations.

```markdown
## PR Review: #<number>
<!-- Or for local branch reviews: -->
## Branch Review: <branch-name> (vs main)

### Summary
Brief overall assessment of the changes (1-2 sentences).

### Code Quality
[Issues and suggestions]

### Infrastructure
[Flag any checklist items from the PyTorch Infrastructure section that apply.
Reference the specific infrastructure the PR should be using.]

### Testing
[Testing adequacy findings — missing OpInfo usage, non-device-generic tests, etc.]

### API Design
[Flag new patterns, internal-access flags, or broader implications if any.]

### Security
[Issues if any]

### Thread Safety
[Threading concerns if any]

### Backward Compatibility
[BC concerns if any]

### Performance
[Performance concerns if any]

### Recommendation
**Approve** / **Request Changes** / **Needs Discussion**

[Brief justification for recommendation]
```

### Specific Comments (Detailed Review Only)

**Only include this section if the user requests a "detailed" or "in depth" review.**

**Do not repeat observations already made in other sections.** This section is for additional file-specific feedback that doesn't fit into the categorized sections above.

When requested, add file-specific feedback with line references:

```markdown
### Specific Comments
- `src/module.py:42` - Consider extracting this logic into a named function for clarity
- `test/test_feature.py:100-105` - Missing test for error case when input is None
- `torch/nn/modules/linear.py:78` - This allocation could be moved outside the loop
```

## Key Principles

1. **Investigate, don't guess** - When uncertain whether a checklist item applies, spawn a sub-agent to read the relevant infrastructure code. A reviewer who guesses wrong provides negative value. A reviewer who investigates and reports findings provides immense value.
2. **Every line matters** - A single missing `C10_CUDA_KERNEL_LAUNCH_CHECK()`, a single `weights_only=False`, a single missing Composite dispatch key — each of these is a real bug that affects real users. Do not skip lines.
3. **No repetition** - Each observation appears in exactly one section. Never repeat the same issue, concern, or suggestion across multiple sections. If an issue spans categories (e.g., a security issue that also affects performance), place it in the most relevant section only.
4. **Focus on what CI cannot check** - Don't comment on formatting, linting, or type errors
5. **Be specific** - Reference file paths and line numbers. Every finding should point to a concrete line in the diff.
6. **Be actionable** - Provide concrete suggestions with the right infrastructure to use, not vague concerns. If flagging a missing pattern, name the function/class/file the author should use.
7. **Be proportionate** - Minor issues shouldn't block, but note them
8. **Assume competence** - The author knows PyTorch; explain only non-obvious context. The value of this review is in catching infrastructure patterns the author may not know about, not in explaining basic programming.

## Files to Reference

When reviewing, consult these project files for context. **Spawn sub-agents to read these** rather than relying on memory — the files change frequently:
- `CLAUDE.md` - Coding style philosophy and testing patterns
- `CONTRIBUTING.md` - PR requirements and review process
- `torch/testing/_internal/common_utils.py` - Test patterns and utilities
- `torch/testing/_internal/opinfo/core.py` - OpInfo test framework
- `aten/src/ATen/native/native_functions.yaml` - Operator declarations (for checking tags, dispatch keys, structured kernels)
- `tools/autograd/derivatives.yaml` - Backward formulas (for checking if an op should register here)
- `aten/src/ATen/native/tags.yaml` - Operator semantic tags
