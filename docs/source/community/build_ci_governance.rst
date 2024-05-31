PyTorch Governance | Build + CI
===============================

How to Add a New Maintainer
---------------------------

For the person to be a maintainer, a person needs to:

* Land at least six commits to the related part of the PyTorch repository
* At least one of these commits must be submitted in the last six months

To add a qualified person to the maintainers' list, please create
a PR that adds a person to the `persons of interests <https://pytorch.org/docs/main/community/persons_of_interest.html>`__ page and
`merge_rules <https://github.com/pytorch/pytorch/blob/main/.github/merge_rules.yaml>`__ files. Current maintainers will cast their votes of
support. Decision criteria for approving the PR:

* Not earlier than two business days passed before merging (ensure the majority of the contributors have seen it)
* PR has the correct label (`module: ci`)
* There are no objections from the current maintainers
* There are at least three net *thumbs up* from current maintainers (or all maintainers vote *thumbs up* when the module has less than 3 maintainers).
