# .github/workflows

**Gotcha — moving scripts breaks in-flight PRs.** Because `checkout-pytorch` checks out `pull_request.head.sha` instead of the merge commit, moving or renaming any script referenced by a workflow `run:` command on `main` will break PRs that branched before the move. See [workflow-checkout-version-mismatch.md](../docs/workflow-checkout-version-mismatch.md) for full analysis.
