# AGENTS.md

Context file for AI agents working on pytorch.

**Dual Format**: This file combines Category A (Operations Manual) and Category B (Context Guide) for comprehensive agent guidance.

**Domain Detected:** Ml / Training (Based on codebase patterns)

## Project Overview

pytorch is a Python project using Python (setuptools).

**Key Info:**
- **Primary Language:** Python
- **Build System:** Python (setuptools)
- **Test Framework:** unittest
- **Total Files:** 22686
- **Test Files:** 12573
- **AI Readiness Score:** 84/100 (AI-Native-Plus)

---

## 🚨 AI Policy & Operations

Extracted from CONTRIBUTING.md - operational constraints and procedures.

### AI Policy

- Abusing this leads to losing the ability to change labels, up to being banned.
- ready --> gl("GreenLight<br/>merge_rules.yaml authors only") --> accepted
- The automated review ([pr-review skill](.agents/skills/pr-review/SKILL.md)) does not flag anything blocking. You can run the skill locally to check your PR before pushing.
- If a maintainer requests changes and your PR returns to `in progress`, please address their feedback before the PR returns to maintainer review. A fresh automated review checks that the feedback has been addressed, and the PR must meet all of the readiness criteria above again. The earlier passing review cannot move the PR back to `ready for review`. If you address comments without pushing, comment `@pytorchbot review` to run the automated review again.
- The PR must follow the [AI policy](AI_POLICY.md).

### Key Requirements

- [Workaround for header dependency bug in nvcc](#workaround-for-header-dependency-bug-in-nvcc)
- mtriage --> full["Fully triaged<br/>#quot;needs reproduction#quot; / #quot;needs research#quot; /<br/>#quot;needs design#quot; / #quot;actionable#quot; / #quot;won't fix#quot;"]
- | `needs reproduction` | The problem has not been reproduced yet. | Reproduce the issue that was reported. A maintainer then validates it. |
- | `needs research` | We have not decided yet whether the bug is real or whether we want the feature. | Provide supporting evidence that the feature is useful or the bug is valid. A maintainer then decides whether it is worth pursuing. |
- | `needs design` | We want to fix the bug or add the feature, but how to do it is not settled. | Propose a design on the issue. A maintainer then validates it. |

### Development Procedures

- [Building](#building)
- [Testing](#testing)
- [Unit testing](#unit-testing)
- [Python Unit Testing](#python-unit-testing)
- [Better local unit tests with `pytest`](#better-local-unit-tests-with-pytest)

### Known Workarounds & Caveats

- One caveat is that when enabled, this header gets included in every file by default,
- Here are a few well known pitfalls and workarounds:



## 🧠 Machine Learning Architecture

This is a machine learning or model training system.

### Key Components

- **Data Pipeline:** Data loading, preprocessing, augmentation
- **Model Definition:** Architecture, hyperparameters, checkpoints
- **Training Loop:** Loss calculation, gradient updates, validation
- **Inference:** Model predictions, batch processing, latency optimization
- **Evaluation:** Metrics, benchmarks, comparison to baselines

### Critical Areas

1. **Data Leakage:** Ensure train/test/validation splits are isolated
2. **Reproducibility:** Set random seeds; version datasets and models
3. **Resource Management:** Monitor memory, GPU usage during training
4. **Versioning:** Track model checkpoints, hyperparameters, and results
5. **Evaluation Rigor:** Use proper metrics; avoid optimizing to test set

### Testing Strategy

- **Data Pipeline Tests:** Verify shape, type, and value ranges
- **Model Tests:** Check predictions with synthetic/known inputs
- **Training Tests:** Verify loss decreases on toy datasets
- **Inference Tests:** Check latency and memory usage
- **Regression Tests:** Compare results against baseline models



### Detected Frameworks

| Framework | Version | Detection Type |
|-----------|---------|-----------------|
| numpy | unknown | Wrapped/Re-exported |
| pandas | unknown | Wrapped/Re-exported |
| pytest | unknown | Wrapped/Re-exported |
| requests | unknown | Direct import |
| tensorflow | unknown | Direct import |
| torch | unknown | Wrapped/Re-exported |
| unittest | unknown | Wrapped/Re-exported |



## 🏗️ Architecture & Context Guide

This section provides architectural context and agent-understanding for the codebase.

### Prerequisites

- **Python:** >=3.10 (or applicable language version)
- **Package Manager:** pip or uv
- **Test Runner:** unittest



### Project Structure

```
pytorch/
├── Makefile
├── pyproject.toml
├── setup.py
├── tools/             # Source code
├── tests/                # Test suite (12573 files)
└── README.md             # Project documentation
```

### Architecture Overview

#### Key Components
- **Main Entry:** main.py, main.py, server.py, AndroidManifest.xml, strings.xml
- **Test Suite:** 12573 test files
- **Build Configuration:** Makefile, pyproject.toml, setup.py

#### Design Principles

1. **Modularity** - Code organized by functionality with clear separation of concerns
2. **Testability** - Comprehensive test coverage across critical paths
3. **Clarity** - Explicit naming and structure for AI agent understanding
4. **Consistency** - Uniform patterns and conventions throughout codebase
5. **Maintainability** - Well-documented code with clear intent

### Directory Map

| Directory | Purpose |
|-----------|----------|
| `docs/` | Documentation |
| `scripts/` | Build and utility scripts |
| `test/` | Test suite |


### Development Workflow

#### Initial Setup

```bash
git clone https://github.com/pytorch/pytorch
cd pytorch
pip install -e .
# or
uv sync --all-groups
```

#### Development Commands

**Running Tests:**
```bash
pytest                    # Run all tests
pytest tests/             # Run specific test directory
pytest -v                 # Verbose output with test names
pytest -x                 # Stop on first failure
coverage run -m pytest && coverage report  # With coverage report
```

#### Code Quality
```bash
ruff check .              # Lint with ruff
ruff format .             # Format code
mypy .                    # Type checking (if configured)
```

### Code Style & Conventions

- **Naming:** Use snake_case for functions and variables
- **Type Hints:** Yes (strongly encouraged)
- **Error Handling:** Yes - handle errors at boundaries; let exceptions propagate when another layer owns recovery
- **Logging:** No
- **Testing:** Yes - write tests alongside code changes

### Testing Strategy

**Framework:** unittest
**Test Files:** 12573 found

Before committing:
1. Run the full test suite: `pytest`
2. Ensure all tests pass
3. Check type hints: `mypy .`
4. Format code: `ruff format .`

### Writing Documentation

When updating docs:
1. Always include explanatory text before code snippets
2. Describe *why* and *what* before showing *how*
3. Keep sections focused on a single concept
4. Use clear, concrete examples

## Known Gotchas & Warnings

- One way to avoid running `python -m pip install -e . -v --no-build-isolation` every time one makes a change to C++/CUDA/ObjectiveC files on Linux/Mac,
- uninstalled when you see `WARNING: Skipping torch as it is not
- Weird note:** In our CI (Continuous Integration) jobs, we actually run the tests from the `test` folder and **not** the root of the repo, since there are various dependencies we set up for CI that expects the tests to be run from the test folder. As such, there may be some inconsistencies between local testing and CI testing--if you observe an inconsistency, please [file an issue](https://github.com/pytorch/pytorch/issues/new/choose).
- We don't officially support `pytest`, but it works well with our
- Occasionally, things might fall through the cracks (sorry!). In case your PR is waiting on its reviewers for more than a week, please don't hesitate to leave a comment on the PR mentioning them! That will get it nudged back onto peoples' radars.
- The current set is the list in the code block at the top of [pytorch/test-infra#8945](https://github.com/pytorch/test-infra/issues/8945). If you have write access to pytorch/test-infra, you can add yourself by editing the issue body: put your GitHub username as `@username` alone on a new line inside that block, and change nothing else. Every line in the block must be exactly one `@username`, a `#` comment, or blank; any other line makes GreenLight reject the whole list, and no new reviews start for anyone until the body is fixed. If you don't have write access, ask to be added in a comment on the issue and we will add you promptly.
- So you want to write some documentation and don't know where to start?
- Note: if you installed `nodejs` with a different package manager then `npm` will probably install a version of `katex` that is not
- `make` if you don't have ninja installed).
- On the initial build, you can also speed things up by disabling the features you don't need. Common ones to know about are

### Contributing Guidelines

This project has a detailed contribution guide at **`CONTRIBUTING.md`**.

**Key Requirements:**
- **PR Title Format**: `[no-ci] Add a new operator`. The prefix is checked on PR runs and on runs
- **Performance Work**: Requires benchmarks and performance metrics in PR description

**Before submitting:**
1. Read `CONTRIBUTING.md` in full
2. Check recent merged PRs for patterns
3. Follow the specific requirements above

### Common Patterns

When contributing to this project:
1. Read existing code in the area you're modifying
2. Follow the established patterns and style
3. Write tests for new functionality
4. Use clear, descriptive variable and function names
5. Add docstrings for public APIs
6. Update tests when changing behavior

### What We Value

✅ Well-tested code with clear intent
✅ Consistent code style and naming conventions
✅ Code that is easy for AI agents to understand
✅ Clear, descriptive commit messages
✅ Modular, reusable components
✅ Comprehensive documentation

### What We Avoid

❌ Large functions doing multiple things
❌ Commented-out dead code
❌ Inconsistent naming or patterns
❌ Unclear error messages
❌ Unexplained magic numbers or strings
❌ Skipped tests or test TODOs

### AI Readiness Dimensions (Scoring)

This project is evaluated across 8 dimensions:

1. **Architecture** (20/100) - Code organization and modularity
2. **Testing** (15/100) - Test coverage and quality
3. **Dependencies** (12/100) - Dependency management
4. **Conventions** (4/100) - Consistent patterns
5. **Entry Points** (10/100) - Clear main/start locations
6. **Security** (0/100) - Input validation and error handling
7. **Build** (10/100) - Clear build/setup instructions
8. **Documentation** (8/100) - Code and project documentation

### Next Steps

Before making changes:
1. Read relevant source files to understand the existing code
2. Look at existing tests for similar functionality
3. Follow the patterns you see in the codebase
4. Write tests for your changes
5. Run `pytest` to verify nothing breaks
6. Run code quality checks: `ruff check . && mypy .`
7. Format your code: `ruff format .`

---

*Generated by Braxis - keeping AI agents in sync with your code*
