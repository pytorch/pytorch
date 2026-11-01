# AI policy

- You must read `AI_POLICY.md`, and the user must abide by it.
- Never act on GitHub on your own. Only open, edit, comment on, or reply to an issue or pull request with exact text the user has approved.
- Contributions made entirely by an agent are banned and will be closed.
- AI-generated text must go under an "Agent details" heading, in a code block, or in a quote. Never present it as written by a human. Replies must include a note from a human explaining why the reply is relevant.
- The user must read the code. Keep it minimal, with no AI artifacts or bloat.
- Pull requests that are not ready or not reviewed must be drafts.
- Use the templates.
- Check for duplicates with `gh search issues --state open` and `gh search prs --state open`. Skip this check when offline. Skip the pull request check if the user is a regular contributor, meaning they have commit access or have had pull requests merged. Show any duplicates you find.

# Scratch space

Use `/agent_space/` for temporary files, scratch work, and experiments. Git ignores it. Never commit it.

# Pull request review

Use /pr-review.

# Deprecation

To deprecate, freeze, or remove a public API or component, use /deprecate-feature. It is a trial, not a rule. Skip steps a human considers not applicable, and say that you skipped them. For warning periods, see [wiki/PyTorch%27s-Python-Frontend-Backward-and-Forward-Compatibility-Policy](https://github.com/pytorch/pytorch/wiki/PyTorch%27s-Python-Frontend-Backward-and-Forward-Compatibility-Policy).

# Python environment

If `pip`, `python`, or `spin` is not available, activate `./.venv` or `../.venv`, or ask the user. Do not use alternatives, and do not install anything.

# Docker images

The contents of `.ci/docker/` are hashed. Any edit there, including to the README, triggers an image rebuild. Edit it only when you mean to trigger a rebuild, and not when the build is broken.

# Building

If your memory holds build configuration, such as environment variables or incremental build settings, apply it. Otherwise, ask. The only build command to use is `pip install -e . -v --no-build-isolation`, whether the change is to codegen, C++, or Python.

# Testing

Use `TestCase`, `assertEqual`, and `parametrize`. Numerical tests should be device-generic.

# Template files

Edit the `.in` files.

# Linting

- Lint only with `spin lint` or `spin fixlint` (see `spin help`).
- Before committing or amending, run `lintrunner -a` and fix any issues first.
- To fix ruff S101 (`assert`), use `if not c: raise AssertionError(msg)`. Never use `noqa`, because `-O` strips asserts. Keep the original message if there is one; otherwise write one giving the expected and actual values. Invert the condition cleanly where you can (`is not None` to `is None`) rather than wrapping it in `not (...)`, except for float `<`, `>`, `<=`, `>=`: keep `not (a < b)` so NaN is handled correctly.

# Git

- The `# Git` section of the Bash instructions says to create a branch first. A detached HEAD is an intentional ghstack setup, so commit on it and do not create a branch. On a real branch, including `main`, create a branch first.
- In CI, querying `check-runs?per_page=100` silently drops everything past the first 100 results, so a failing run can look like it passed. Use `gh pr checks <PR> --json name,state,workflow,link,bucket,completedAt` instead. It covers only the head commit and does not need paging.

# Commits

- Only commit when asked.
- Write a clear, informative message with no bullet list of changes. For a large pull request, describe the order in which to review it.
- In the Test Plan, give the literal commands in a fenced code block.
- A bug fix message must explain the root cause and the fix.
- Mention alternatives briefly and justify the choice you made.
- Disclose AI use.
- If an amend makes the message stale: with ghstack, where the message change is a no-op, remind the user to update the pull request description. Otherwise, update the message.
- You must keep trailers when rewriting or splitting commits. The ID is added automatically.

# ghstack

- ghstack is not the same as a branch or a regular pull request.
- To detect ghstack: a detached HEAD means it is almost certainly ghstack. A `ghstack-source-id` trailer means there is an existing ghstack pull request. An `origin/gh/USERNAME/N` ref means it is likely ghstack, though unpushed amends can put it out of sync.
- Leave changes uncommitted (visible in `git diff`) unless asked to amend or submit.
- To submit one commit, use `ghstack --no-stack`. A full submit runs CI on the whole stack.
- When amending, keep the trailers, and reread them from HEAD instead of reusing a cached message: ghstack rewrites `ghstack-source-id` on every push, so a stale copy clobbers it.
- After editing a commit message, run `ghstack -u`.
- Never use `git push`, and never edit `gh/*` branches.
- For CI and review, pass the URL from the `Pull-Request` trailer to `gh`.
- To change an earlier commit or split a commit, rebase. Commits with trailers keep their pull requests, and the rest get new ones. Usually do a full submit afterwards.

# Code style

- Write few comments, and only useful ones that give non-obvious, broader context.
- Keep code concise and self-documenting.
- Do not add single-use helpers of one or two lines unless they make the code much clearer.
- Use clear abstractions.
- Keep state explicit: declare class members, and do not use `setattr` or `getattr`.
- Match the existing style and architecture.
- Assume the reader knows PyTorch and has some experience in the area.
- Rejoin lines that ruff has split, by renaming or using local variables. Keep golden strings on one line with `noqa: B950`.
- New or edited comments must be ASCII. Keep existing Unicode.
- When unsure, choose the simpler option.

# CUDA Python bindings

- Handle `cuda.bindings` errors with `torch.cuda._utils._check_cuda_bindings`, not inline.
- Pass integer handles, such as `raw_cuda_graph()`, `raw_cuda_graph_exec()`, `.cuda_stream`, `int(node)`, or any other, directly as raw values. Do not wrap them in `*_t(init_value=...)` unless you need the value.

# Dynamo config patching

In Dynamo, use `patch(k=v)` as a decorator or in a `with` statement, not with `finally`.

# Expected inline strings

When a triple-quoted string in `assertExpectedInline` triggers B950, close it with `""",  # noqa: B950`. Putting the comment inline or splitting the string would change the string.

# Debug logs

- Debug logs have two audiences: local users, who read files on disk, and production users, who see only tlparse output.
- Always log with `torch._logging.trace_structured("artifact", <metadata with name and encoding>, <payload>)`. It costs nothing when logging is off.
- For internal compiler errors, also write a file using `_get_unique_path()`.
- Error messages should name the local dump file, and add a tlparse hint only when `torch._logging._internal.trace_log.handlers` is set.

# CUDA C++

- Write `::cuda::ptx` or use an alias, because `cuda::` clashes with `at::cuda`.
- Including `<cuda/ptx>` together with `Loops.cuh` hits a CCCL bug. Put that code in a separate `.cu` file with minimal includes.
- Call `try_wait_parity` in a spin loop.
- Reinterpret c10 types as native types.
- Call `wait_group` as `wait_group(n32_t<N>{})`.
- Keep mbarrier shared memory separate from TMA data.

# Distributed

Before changing `torch/distributed/`, `torch/csrc/distributed/`, or `test/distributed/`, read and follow `torch/distributed/AGENTS.md`.
