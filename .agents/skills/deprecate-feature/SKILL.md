---
name: deprecate-feature
description: Deprecate, freeze, or remove a PyTorch API or component and track it in the deprecation umbrella issue #201298. Use when adding a deprecation warning, choosing a removal version, removing previously deprecated code, closing an issue because a component is frozen, or auditing overdue "will be removed in X.Y" messages.
---

# Deprecating a Feature in PyTorch

Use one tracking issue for the whole lifecycle of a deprecation, freeze, or removal. File it with the deprecation form (`.github/ISSUE_TEMPLATE/deprecation.yml`) and attach it as a sub-issue of the umbrella https://github.com/pytorch/pytorch/issues/201298. Report any unmet [Done checklist](#done-checklist) item as unfinished work.

## Lifecycle

This follows the [backward compatibility policy](https://github.com/pytorch/pytorch/wiki/PyTorch%27s-Python-Frontend-Backward-and-Forward-Compatibility-Policy) for stable components; if this skill and the wiki disagree, follow the wiki. Prototype and beta components may break without a deprecation period.

1. **Deprecate in release N.** Warn on use, point to the replacement, update docs, and migrate PyTorch's own callers.
2. **Warn for at least two releases** (180 days for nightly users). The earliest removal target is N+2.
3. **Remove** in N+2 or later, once the announced target has arrived.
4. **Silent behavior changes** (e.g. changing a default) must be made loud first: after the warning period, the old behavior raises an error for at least two more releases (180 days for nightly users) before the new behavior is enabled.

N is the release that will ship the change: read `version.txt` (`2.16.0a0` means the change ships in 2.16, so the earliest removal is 2.18).

## Adding a deprecation

The message must state:
- the start version and the removal target as concrete numbers, never "a future release";
- the replacement, or how to keep the current behavior if there is none;
- the tracking issue URL.

Warn at most once per process so users are not spammed.

Python: emit a `FutureWarning` (`DeprecationWarning` is hidden by default outside `__main__`) behind a module-level once flag. The default warnings filter only deduplicates per call site, so an API reached from many places would otherwise warn repeatedly:

```python
_foo_warned = False

def foo(...):
    global _foo_warned
    if not _foo_warned:
        _foo_warned = True
        warnings.warn(
            "`torch.foo` is deprecated since 2.16 and will be removed in 2.18. Use `torch.bar` instead. "
            "See https://github.com/pytorch/pytorch/issues/NNNN",
            FutureWarning,
            stacklevel=2,
        )
```

To also get type-checker diagnostics, add `@typing_extensions.deprecated(msg, category=None)`. Do not pass a warning category to `@deprecated`: it warns on every call, independently of the once flag.

C++: `TORCH_WARN_ONCE(...)`, which surfaces in Python as a `UserWarning`. Do not use `TORCH_WARN` (repeats) or `TORCH_WARN_DEPRECATION` (repeats, and maps to Python `DeprecationWarning`, which is hidden by default).

Also:
- Warn where the user opts into the deprecated behavior (the call or argument), not in hot internal paths.
- Add `.. deprecated:: 2.16` to the docstring.
- Test that the warning fires with the expected text (reset the once flag in the test), and silence it in other tests that still exercise the old path.
- Label the PR `topic: deprecation` so it lands in the release notes.

## Freezing a component

Some components are frozen rather than scheduled for removal (e.g. the C++ frontend `torch::nn`/`torch::optim`). Frozen means: no new features, and bugs are closed as won't-fix unless they are security or crash issues. The component gets a tracking issue (Kind: Frozen component; Removal target and Next step: `none` if no removal is planned); link it when closing issues against the component.

## Tracking issue

1. Search for an existing tracking issue: sub-issues of #201298 and `gh issue list -R pytorch/pytorch -l "module: deprecation" --search "<api>"`.
2. If none exists, draft one with the form's fields: API or component, Kind, Deprecated in, Removal target, Next step, Replacement and migration, Owner, PRs. Render each field as a `### <label>` heading followed by its value, as the form does, so automation can parse it. Title: `[deprecation] <api>`.
3. Get user approval of the exact title and body before creating the issue, and of the exact change before editing any existing issue (see `AI_POLICY.md`). Create it with the `module: deprecation` label. Add `slowroll` only for a staged BC rollout behind an opt-in flag (e.g. #191236).
4. Attach it as a sub-issue of #201298. The API takes the issue's internal `id`, not its number:

   ```bash
   id=$(gh api repos/pytorch/pytorch/issues/<N> --jq .id)
   gh api -X POST repos/pytorch/pytorch/issues/201298/sub_issues -F sub_issue_id="$id"
   ```

5. Keep `Next step` current, e.g. `remove in 2.18`. Milestones are assigned from it at each branch cut, so do not set future milestones by hand.

## Removing a deprecated feature

1. Find the start version in the tracking issue, the warning, or `git log -S '<warning text>'`. Verify the warning has shipped in at least two releases and that the announced removal target has arrived.
2. Remove the implementation, the warning, and its tests. Search for remaining references, including C++, `.pyi.in` stubs, `docs/source`, and `torch/overrides.py`. Where docs remain (e.g. for a behavior change), add `.. versionchanged:: <version>` with migration guidance.
3. Label the PR `topic: bc breaking` and describe the migration in the release-notes section.
4. If steps remain (e.g. enabling new behavior), update `Next step` on the tracking issue; otherwise the PR closes it.

## Done checklist

- [ ] A new warning names the start version, the removal target, and the tracking issue URL, and fires at most once per process.
- [ ] The tracking issue is labeled `module: deprecation`, has an up-to-date `Next step`, and is a sub-issue of #201298.
- [ ] The PR body contains `Part of #<tracking issue>`, or `Fixes #<tracking issue>` for the final step, and the PR is labeled `topic: deprecation` (or `topic: bc breaking` for a removal).

## Auditing overdue removals

List warnings whose removal target has passed, then compare the versions against `version.txt`:

```bash
cat version.txt
git grep -nIE "(removed|deleted|dropped|become an error) (in|by|after) (PyTorch |version |release |v)?[0-9]+\.[0-9]+" -- torch aten c10
```

Every hit at or below the current version is overdue: remove it, or retarget it with a reason. Each one needs a tracking issue under #201298.
