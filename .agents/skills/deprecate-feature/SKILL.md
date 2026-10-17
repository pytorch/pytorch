---
name: deprecate-feature
description: Deprecate, freeze, or remove a PyTorch API or component and track it in the deprecation umbrella issue #201298. Use when adding a deprecation warning, choosing a removal version, removing previously deprecated code, closing an issue because a component is frozen, or auditing overdue "will be removed in X.Y" messages.
---

# Deprecating a Feature in PyTorch

Every deprecation needs a start release, a removal target, and an entry in the umbrella issue https://github.com/pytorch/pytorch/issues/201298. Deprecations without a target and a tracker entry tend to stay forever (main still warns about removals planned for 1.6).

## Lifecycle

1. **Deprecate (release N).** Warn on use, point to the replacement, update docs, and stop using the API inside PyTorch itself.
2. **Keep warning for at least two releases.** The earliest removal target is N+2.
3. **Remove (release >= N+2).** Delete the code, its tests, and its docs. For a behavior change (rather than a plain removal), turn the warning into an error for one release before enabling the new behavior so programs do not silently change results.

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

C++: `TORCH_WARN_ONCE(...)` (or `TORCH_WARN_DEPRECATION(...)`) with the same message shape.

The message must state:
- the start version and the removal version as concrete numbers (`since 2.15 ... removed in 2.17`), never "a future release";
- the replacement, or how to keep the current behavior if there is none.

Also:
- Add `.. deprecated:: 2.15` to the docstring.
- Add a test that the warning fires (`self.assertWarnsRegex(FutureWarning, ...)`), and silence it in other tests that still exercise the old path.
- Label the PR `topic: deprecation` so it lands in the release notes.

## Freezing a component

Some components are frozen rather than scheduled for removal (e.g. the C++ frontend `torch::nn`/`torch::optim`). Frozen means: no new features, bugs closed as won't-fix unless they are security or crash issues. Make sure the component has an entry in #201298 with the freeze date and, if known, a removal target, and link that entry when closing issues against it.

## Tracking in #201298

Every deprecation, freeze, and removal must be reflected in the umbrella:

- **New deprecation:** add a checklist item with the API, start version, removal target, owner, and PR/issue link, for example
  `- [ ] torch.foo - deprecated 2.15, remove in 2.17 - @owner - #12345`
- **Removal landed:** check the item off and link the removal PR.
- **Target slipped:** update the target and give the reason in the item.

Do not edit the issue or post comments yourself. Draft the exact text and have the user review and apply it (see `AI_POLICY.md`).

## Removing a deprecated feature

1. Check the start version in the warning, the docstring, or `git log -S '<warning text>'`. Removal is allowed only once `version.txt` is at least start + 2.
2. Delete the implementation, the warning, its tests, and its docs. Search for remaining references, including C++, `.pyi.in` stubs, `docs/source`, and `torch/overrides.py`.
3. Label the PR `topic: bc breaking` and describe the migration in the release-notes section of the PR.
4. Update the item in #201298.

## Auditing overdue removals

List warnings whose removal target has passed, then compare the versions against `version.txt`:

```bash
cat version.txt
git grep -nIE "(removed|deleted|dropped|become an error) (in|by|after) (PyTorch |version |release |v)?[0-9]+\.[0-9]+" -- torch aten c10
```

Every hit with a version at or below the current one is overdue: either remove it or retarget it with a reason. Make sure each one is listed in #201298.
