---
name: deprecate-feature
description: Deprecate, freeze, or remove a PyTorch API or component and track it in the deprecation umbrella issue #201298. Use when adding a deprecation warning, choosing a removal version, removing previously deprecated code, closing an issue because a component is frozen, or auditing overdue "will be removed in X.Y" messages.
---

# Deprecating a Feature in PyTorch

Every deprecation, freeze, and removal is tracked by its own issue, filed with the deprecation issue form (`.github/ISSUE_TEMPLATE/deprecation.yml`) and attached as a sub-issue of the umbrella https://github.com/pytorch/pytorch/issues/201298. Deprecations without a target and a tracking issue tend to stay forever (main still warns about removals planned for 1.6).

The work is not done until every item in the [Done checklist](#done-checklist) holds. If one is missing, report it as unfinished instead of finishing.

## Lifecycle

This follows the [backward compatibility policy](https://github.com/pytorch/pytorch/wiki/PyTorch%27s-Python-Frontend-Backward-and-Forward-Compatibility-Policy) for stable components. Prototype and beta components may break without a deprecation period.

1. **Deprecate (release N).** Warn on use, point to the replacement, update docs, and stop using the API inside PyTorch itself.
2. **Keep warning for at least two releases** (and 180 days of nightlies). The earliest removal target is N+2.
3. **Remove (release >= N+2).** Delete the code, its tests, and its docs.
4. **Silent behavior changes** (e.g. changing a default) must be made loud first: after the warning period, the old behavior raises an error for another two releases (N+2 to N+4) before the new behavior is enabled.

Release N is the release that will ship the change: read `version.txt` (e.g. `2.15.0a0` means the change ships in 2.15, so the earliest removal is 2.17).

## Adding a deprecation

Python, prefer `typing_extensions.deprecated` with `FutureWarning` (user-facing; `DeprecationWarning` is hidden by default outside `__main__`):

```python
from typing_extensions import deprecated

@deprecated(
    "`torch.foo` is deprecated since 2.15 and will be removed in 2.17. Use `torch.bar` instead.",
    category=FutureWarning,
)
def foo(...): ...
```

For a deprecated argument or behavior inside a function, use `warnings.warn(msg, FutureWarning, stacklevel=2)` so the warning points at user code.

C++: `TORCH_WARN_ONCE(...)` with the same message shape.

Do not spam users. Warn at most once per process:
- C++: use `TORCH_WARN_ONCE`, not `TORCH_WARN`.
- Python: the default filter shows a warning once per call site, which still repeats when the deprecated path is reached from many places (or under filters that reset the registry). For APIs that can be hit repeatedly, such as per-op, per-step, or per-module calls, guard the warning with a module-level flag so it fires once per process.
- Warn where the user opts into the deprecated behavior (the call or argument), not in hot internal paths.
- PyTorch itself must not trigger the warning: migrate internal callers so users only see warnings for their own code.

The message must state:
- the start version and the removal version as concrete numbers (`since 2.15 ... removed in 2.17`), never "a future release";
- the replacement, or how to keep the current behavior if there is none;
- the tracking issue (`See https://github.com/pytorch/pytorch/issues/NNNN`).

Also:
- Add `.. deprecated:: 2.15` to the docstring.
- Add a test that the warning fires (`self.assertWarnsRegex(FutureWarning, ...)`), and silence it in other tests that still exercise the old path.
- Label the PR `topic: deprecation` so it lands in the release notes.

## Freezing a component

Some components are frozen rather than scheduled for removal (e.g. the C++ frontend `torch::nn`/`torch::optim`). Frozen means: no new features, bugs closed as won't-fix unless they are security or crash issues. The component gets a tracking issue (Kind: Frozen component, Removal target: `none` if unplanned); link it when closing issues against the component.

## Tracking issue

1. Search for an existing tracking issue: sub-issues of #201298 and `gh issue list -R pytorch/pytorch -l "module: deprecation" --search "<api>"`.
2. If none exists, draft one using the fields of `.github/ISSUE_TEMPLATE/deprecation.yml`: API or component, Kind, Deprecated in, Removal target, Next step, Replacement and migration, Owner, PRs. Render each field as a `### <label>` heading followed by its value, matching what the form produces, so automation can parse it. Title: `[deprecation] <api>`.
3. Show the exact title and body to the user and wait for approval before creating anything (see `AI_POLICY.md`). Then create it with labels `module: deprecation` and `slowroll`.
4. Attach it as a sub-issue of #201298. The API takes the issue's internal `id`, not its number:

   ```bash
   id=$(gh api repos/pytorch/pytorch/issues/<N> --jq .id)
   gh api -X POST repos/pytorch/pytorch/issues/201298/sub_issues -F sub_issue_id="$id"
   ```

5. Keep `Next step` current: it holds the next action and the release it ships in (e.g. `remove in 2.17`). Milestones are assigned from it at each branch cut, so do not set future milestones by hand.

## Done checklist

- [ ] Warning text names the start version, the removal version (>= start + 2), and the tracking issue URL.
- [ ] Tracking issue exists, is labeled `module: deprecation` and `slowroll`, has an up-to-date `Next step`, and is a sub-issue of #201298.
- [ ] PR body contains `Part of #<tracking issue>`; PR is labeled `topic: deprecation` (or `topic: bc breaking` for a removal).

## Removing a deprecated feature

1. Check the start version in the warning, the docstring, or `git log -S '<warning text>'`. Removal is allowed only once `version.txt` is at least start + 2.
2. Delete the implementation, the warning, its tests, and its docs. Search for remaining references, including C++, `.pyi.in` stubs, `docs/source`, and `torch/overrides.py`.
3. Label the PR `topic: bc breaking`, describe the migration in the release-notes section, and reference the tracking issue (`Fixes #<N>` for the final step, otherwise `Part of #<N>`).
4. If steps remain (e.g. enabling new behavior), update `Next step` on the tracking issue (after user approval).

## Auditing overdue removals

List warnings whose removal target has passed, then compare the versions against `version.txt`:

```bash
cat version.txt
git grep -nIE "(removed|deleted|dropped|become an error) (in|by|after) (PyTorch |version |release |v)?[0-9]+\.[0-9]+" -- torch aten c10
```

Every hit with a version at or below the current one is overdue: either remove it or retarget it with a reason. Make sure each one has a tracking issue under #201298.
