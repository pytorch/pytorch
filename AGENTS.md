# Repository Guidelines

## Project Structure & Module Organization
- `torch/` holds the public Python API, autograd, distributed, and JIT code; `c10/` provides core runtime utilities shared across backends.
- `aten/` contains C++/CUDA operator implementations and code generation; `caffe2/` is kept for legacy integration.
- `test/` hosts Python unit and integration tests; `aten/src/ATen/test/` and `caffe2/test/` cover lower-level pieces.
- `tools/` includes build helpers, linters, and the test runner; `docs/` contains Sphinx sources; `benchmarks/` stores perf suites; `third_party/` vendors dependencies.

## Build, Test, and Development Commands
- Create a Python dev env: `make setup-env` (or `make setup-env-cuda` / `-rocm`), then `source venv/bin/activate`.
- Editable install for Python changes: `python -m pip install -e . -v --no-build-isolation`. Use `python setup.py clean` if builds get stale.
- Lint consistently: `lintrunner -a` auto-applies formatters (clang-format, yapf/black, etc.). Run it before sending a PR.
- Run only focused tests: `python test/test_torch.py TestTorch.test_dir` or `python test/run_test.py --include test_nn -v`. Run the single case relevant to your change.
- For C++/CUDA smoke checks, `python test/run_test.py --cpp` runs the compiled tests that cover native code paths.

## Coding Style & Naming Conventions
- Python follows PEP 8; prefer type hints where practical. Use snake_case for functions/vars, CamelCase for classes, and keep imports ordered by stdlib/third_party/local.
- C++/CUDA matches existing namespaces (`at::`, `c10::`), uses CamelCase types and snake_case functions, and prefers `const`/`constexpr` for invariants. Keep headers light and includes ordered.
- Tests live near the code they exercise; test files/functions start with `test_`. Use utilities in `torch.testing._internal.common_utils` for deterministic seeds and temporary files.

## Testing Guidelines
- When fixing a bug, first craft a minimal standalone repro script to prove the failure, then port the assertion into the nearest maintained test file once the fix is ready.
- Favor smallest inputs that hit the code path; avoid GPU-only cases unless required and gate with `TEST_WITH_*` when needed.
- Document the exact command you ran (from the section above) in PRs; add targeted regression tests rather than broad integration sweeps.

## Commit & Pull Request Guidelines
- Commit messages mirror the repo’s style: `[Area] concise change summary (#issue)` in imperative mood.
- PRs should explain motivation, link issues, list test commands executed, and note BC/perf impacts. Include screenshots for doc or UX changes and benchmarks for perf-sensitive paths.
- Keep diffs scoped; avoid drive-by refactors in unrelated subsystems. Avoid touching submodules or adding third-party code without maintainer buy-in.
