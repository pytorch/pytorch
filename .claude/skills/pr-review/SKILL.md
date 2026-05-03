---
name: pr-review
description: Review PyTorch pull requests for code quality, test coverage, security, and backward compatibility. Use when reviewing PRs, when asked to review code changes, or when the user mentions "review PR", "code review", or "check this PR".
---

# PyTorch PR Review Skill

Review PyTorch pull requests focusing on what CI cannot check: code quality, test coverage adequacy, security vulnerabilities, and backward compatibility.

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

1. **Only report problems** — The review output must contain only issues, concerns, and actionable suggestions. Do NOT mention things that are done correctly, do NOT praise good decisions, do NOT explain why something is fine. If a section has no problems, omit it entirely. The reader's time is precious — every sentence must point to something that needs fixing or further discussion.
2. **Investigate, don't guess** — When uncertain whether a checklist item applies, spawn a sub-agent to read the relevant code. A reviewer who guesses wrong provides negative value.
3. **Review the design, not just the implementation** — A PR can have perfectly correct implementation of a bad design. Question side-channel communication, on/off private flags, and demand concrete interface documentation for new contracts between components.
4. **Focus on what CI cannot check** — Don't comment on formatting, linting, type errors, or CI failures. Focus on design quality, interface correctness, thread safety, BC implications, test adequacy, and pattern adherence.
5. **Everything is a must-fix** — There are no "nits." If it's worth mentioning, it's worth fixing. Every inconsistency degrades the codebase over time.
6. **Be specific and actionable** — Reference file paths and line numbers. Name the function/class/file the author should use.
7. **Match the immediate context** — Read how similar features are already implemented in the same file. Pattern mismatches within a file are always wrong.
8. **Assume competence** — The author knows PyTorch; explain only non-obvious context.
9. **No repetition** — Each observation appears in exactly one section of the review output.

### Using sub-agents

The review checklist is large. You cannot hold the full context of every infrastructure system in your head. **Spawn sub-agents** to investigate whether checklist items apply: read surrounding code, infrastructure the PR should be using, or tests that should exist. Spawn them in parallel for independent areas. A typical medium PR should spawn 3-8 sub-agents.

## Review Workflow

### Step 1: Understand Context

Before reviewing, build understanding of what the PR touches and why:
1. Identify the purpose of the change from title/description/issue
2. Group changes by type (new code, tests, config, docs)
3. Note the scope of changes (files affected, lines changed)
4. Spawn sub-agents to read the unchanged code surrounding each significantly changed file to understand existing patterns and infrastructure

### Step 2: Deep Review

Go through **every changed line** in the diff and evaluate it against the review checklist in [review-checklist.md](review-checklist.md).

### Step 3: Check Backward Compatibility

Evaluate BC implications per [bc-guidelines.md](bc-guidelines.md). For non-trivial BC questions, spawn a sub-agent to search for existing callers of the modified API.

### Step 4: Formulate Review

Structure your review with actionable feedback organized by category. Every finding should be traceable to a specific line in the diff.

### Step 5: Fact-Check

After drafting the review, spawn a sub-agent per reported issue (in parallel) to independently verify the claim by re-reading the relevant code and surrounding context. Each sub-agent returns **valid**, **invalid**, or **needs rewording**. Drop invalid issues, reword the rest. If unsure, leave the issue with a comment for the author that this low confidence.

## Output Format

Structure your review as follows. **Omit sections where you have no problems to report** — most reviews should only have a few sections. Do not write "No concerns", "Looks good", or any affirmative commentary. Every sentence in the review must identify a problem or request a change.

The Summary section is the one exception: it should briefly state what the PR does (1 sentence) and then state the problems found, or explicitly say no issues were found.

```markdown
## PR Review: #<number>
<!-- Or for local branch reviews: -->
## Branch Review: <branch-name> (vs main)

### Summary
What the PR does (1 sentence), then the overall verdict.

### Code Quality
[Problems only]

### Infrastructure
[Problems only — flag checklist items that are violated]

### Testing
[Problems only — missing tests, wrong patterns, inadequate coverage]

### API Design
[Problems only]

### Security
[Problems only]

### Thread Safety
[Problems only]

### Backward Compatibility
[Problems only]

### Performance
[Problems only]

### Recommendation
**Approve** / **Request Changes** / **Needs Discussion**

Missing tests (new functionality without tests, bug fixes without regression tests) always means **Request Changes**.

[Brief justification — focus on what blocks approval, if anything]
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

## Files to Reference

When reviewing, consult these project files for context — read them rather than relying on memory, as they change frequently:
- `CLAUDE.md` - Coding style philosophy and testing patterns
- `CONTRIBUTING.md` - PR requirements and review process
- `torch/testing/_internal/common_utils.py` - Test patterns and utilities
- `torch/testing/_internal/opinfo/core.py` - OpInfo test framework
- `aten/src/ATen/native/native_functions.yaml` - Operator declarations (for checking tags, dispatch keys, structured kernels)
- `tools/autograd/derivatives.yaml` - Backward formulas (for checking if an op should register here)
- `aten/src/ATen/native/tags.yaml` - Operator semantic tags
