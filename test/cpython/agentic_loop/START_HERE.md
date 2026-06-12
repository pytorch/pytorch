# Fresh Context Bootstrap

Use this file when starting the CPython Dynamo agentic loop in a fresh context.

Minimal user prompt:

```text
Start the CPython Dynamo agentic loop from
test/cpython/agentic_loop/START_HERE.md.
```

Agent instructions:

1. Read these files before taking action:

   ```text
   test/cpython/agentic_loop/README.md
   test/cpython/agentic_loop/agent_manager.md
   test/cpython/agentic_loop/coverage.md
   test/cpython/agentic_loop/CPYTHON_MIRRORING.md
   test/cpython/agentic_loop/cpu_fast_ci_baseline.md
   ```

2. Follow `agent_manager.md` exactly. The active role is manager unless the
   human explicitly asks for non-manager work.

3. Work only the current gate in `coverage.md`. Do not batch gates.

4. Use subagents for implementation and review when available. The manager must
   not directly edit files, stage changes, commit, or run tests.

5. Each completed gate should:

   - remove only the proven CPython Dynamo expected-failure sentinel;
   - add or update focused Dynamo regression coverage when the fix is semantic;
   - update `coverage.md` with exact evidence and next gate status;
   - run the CPU fast validation loop from `cpu_fast_ci_baseline.md`;
   - run `lintrunner -a` before commit;
   - land as exactly one commit.

6. Use `agent_space/` for scratch output. Never commit `agent_space/`, logs,
   caches, or unrelated untracked files.

7. If the top-10 gate list may be stale, follow the regeneration instructions
   in `README.md` before beginning implementation.
