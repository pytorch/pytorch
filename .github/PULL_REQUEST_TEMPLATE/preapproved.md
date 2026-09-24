# Pre-approved PR

Use this template for PRs without a linked "actionable" issue, when the change was agreed on with a maintainer beforehand.

Before submitting, please review:
- [PR lifecycle](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md#pr-lifecycle) in the contributing guide
- [AI-Assisted Development](https://github.com/pytorch/pytorch/blob/main/AI_POLICY.md) policy

---

## Approved by

@<!-- Maintainer handle who agreed to this PR. -->

## Summary

<!-- Point to where the pre-approval discussion happened. If the discussion happened in a private channel, provide a couple paragraphs describing the problem and solution. -->

## Checklist

- [ ] Passes lint (`spin fixlint`)
- [ ] Added/updated tests
- [ ] Updated documentation (if applicable)
- [ ] Included benchmark results (for PRs impacting perf)

## BC-breaking?

<!-- If this change breaks backward compatibility, describe the impact and migration path. Otherwise, write "No". -->
