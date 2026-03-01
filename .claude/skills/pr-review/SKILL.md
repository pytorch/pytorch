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

When invoked via workflow, PR data is passed as context. The PR number or diff will be available in the prompt.

## Review Workflow

### Step 1: Fetch PR Information

For local mode, use `gh` commands to get:
1. PR metadata (title, description, author)
2. List of changed files
3. Full diff of changes
4. Existing comments/reviews
5. Fetch associated issue information when applicable

### Step 2: Analyze Changes

Read through the diff systematically:
1. Identify the purpose of the change from title/description/issue
2. Group changes by type (new code, tests, config, docs)
3. Note the scope of changes (files affected, lines changed)

### Step 3: Deep Review

Perform thorough line-by-line analysis using the review checklist. See [review-checklist.md](review-checklist.md) for detailed criteria covering:
- Code quality and design
- Testing adequacy
- Security considerations
- Thread safety and concurrency (Python, C++, CPython C API, NoGIL)
- Performance implications
- Any behavior change not expected by author

### Step 4: Check Backward Compatibility

Evaluate BC implications. See [bc-guidelines.md](bc-guidelines.md) for:
- What constitutes a BC-breaking change
- Required deprecation patterns
- Common BC pitfalls

### Step 5: Formulate Review

Structure your review with actionable feedback organized by category.

## Review Areas

| Area | Focus | Reference |
|------|-------|-----------|
| Code Quality | Abstractions, patterns, complexity | [review-checklist.md](review-checklist.md) |
| API Design | New patterns, flag-based access, broader implications | [review-checklist.md](review-checklist.md) |
| Testing | Coverage, patterns, edge cases | [review-checklist.md](review-checklist.md) |
| Security | Injection, credentials, input handling | [review-checklist.md](review-checklist.md) |
| Performance | Regressions, device handling, memory | [review-checklist.md](review-checklist.md) |
| Thread Safety | Data races, GIL assumptions, NoGIL, CPython C API | [review-checklist.md](review-checklist.md) |
| BC | Breaking changes, deprecation | [bc-guidelines.md](bc-guidelines.md) |

## Output Format

Structure your review as follows:

```markdown
## PR Review: #<number>
<!-- Or for local branch reviews: -->
## Branch Review: <branch-name> (vs main)

### Summary
Brief overall assessment of the changes (1-2 sentences).

### Code Quality
[Issues and suggestions, or "No concerns" if none]

### API Design
[Flag new patterns, internal-access flags, or broader implications if any. Otherwise omit this section.]

### Testing
- [ ] Tests exist for new functionality
- [ ] Edge cases covered
- [ ] Tests follow PyTorch patterns (TestCase, assertEqual)
[Additional testing feedback]

### Security
[Issues if any, or "No security concerns identified"]

### Thread Safety
[Threading concerns if any, or "No thread safety concerns"]

### Backward Compatibility
[BC concerns if any, or "No BC-breaking changes"]

### Performance
[Performance concerns if any, or "No performance concerns"]

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

1. **No repetition** - Each observation appears in exactly one section. Never repeat the same issue, concern, or suggestion across multiple sections. If an issue spans categories (e.g., a security issue that also affects performance), place it in the most relevant section only.
2. **Focus on what CI cannot check** - Don't comment on formatting, linting, or type errors
3. **Be specific** - Reference file paths and line numbers
4. **Be actionable** - Provide concrete suggestions, not vague concerns
5. **Be proportionate** - Minor issues shouldn't block, but note them
6. **Assume competence** - The author knows PyTorch; explain only non-obvious context

## Files to Reference

When reviewing, consult these project files for context:
- `CLAUDE.md` - Coding style philosophy and testing patterns
- `CONTRIBUTING.md` - PR requirements and review process
- `torch/testing/_internal/common_utils.py` - Test patterns and utilities
- `torch/testing/_internal/opinfo/core.py` - OpInfo test framework
