---
name: deprecate-feature
description: Deprecate, freeze, or remove PyTorch APIs; choose a removal version, handle won't-fix frozen components, or audit overdue removals.
---

# Deprecations

This process is provisional; we are evaluating it in practice. Human judgment wins: skip steps that do not apply. New APIs that are technically public but barely used may be deprecated or removed faster at the human's discretion.

The [BC policy](https://github.com/pytorch/pytorch/wiki/PyTorch%27s-Python-Frontend-Backward-and-Forward-Compatibility-Policy) is the source of truth:
- Read release N from `version.txt`.
- Warn for at least two releases (180 days for nightly users); earliest removal is N+2.

## Deprecate or freeze

- State the start version, removal target, replacement (or how to retain current behavior), and tracking issue URL in the warning.
- Warn once per process. Python: `FutureWarning` behind a module-level flag:

  ```python
  _warned = False

  def foo():
      global _warned
      if not _warned or torch.is_warn_always_enabled():
          warnings.warn(message, FutureWarning, stacklevel=2)
          _warned = True
  ```

- Use `@deprecated` only with `category=None` for type-checker diagnostics.
- C++: prefer `TORCH_WARN_ONCE`; `TORCH_WARN_DEPRECATION` repeats and maps to hidden `DeprecationWarning` (#72948).
- Add docstring `.. deprecated:: <N>`, migrate internal callers, test the warning (under `set_warn_always_context(True)`), and label the PR `topic: deprecation`.
- Frozen components also need tracking issues that record the freeze decision; link it when closing issues against the component. Use Kind: Frozen component, with Removal target and Next step: `none` if no removal is planned.

## Track

- Reuse or create one issue per item via `.github/ISSUE_TEMPLATE/deprecation.yml`. Keep its `### <field>` headings.
- Get user approval of the exact text before creating or editing any issue (`AI_POLICY.md`). Label it `module: deprecation`; use `slowroll` only for staged opt-in rollouts (#191236).
- Attach it as a sub-issue of #201298 using its internal **id**, not number:

  ```bash
  id=$(gh api repos/pytorch/pytorch/issues/<N> --jq .id)
  gh api -X POST repos/pytorch/pytorch/issues/201298/sub_issues -F sub_issue_id="$id"
  ```

- Keep `Next step` current (e.g. `remove in 2.18`).
- Link PRs with `Part of #N`, or `Fixes #N` for the final step.

## Remove or audit

- Check that two warning releases shipped and the removal target arrived.
- Remove code, warnings, tests, and references, including `.pyi.in`, docs, and `torch/overrides.py`. Use `.. versionchanged:: <version>` with migration guidance where docs remain; label the PR `topic: bc breaking`.
- Audit targets against `version.txt`; remove overdue items or retarget with a reason, tracking each under #201298:

  ```bash
  git grep -nIE "(removed|deleted|dropped|become an error) (in|by|after) (PyTorch |version |release |v)?[0-9]+\.[0-9]+" -- torch aten c10
  ```
