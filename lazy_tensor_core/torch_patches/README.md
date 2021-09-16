# Guidelines For Patch File Names

Files with extension '.diff' are consider as git patches by apply script.

A file for PyTorch PR _N_ needs to be named 'N.diff'.

Patch files which are not related to PyTorch PRs, should begin with an 'X' character,
followed by a two digit number, followed by a dash ('-'), a name, and '.diff'.
Example:

```
X10-optimizer.diff
```

Patch file are alphabetically ordered, so PyTorch PR patches are always applied
before the non PyTorch ones.


There's a special file `torch_patches/.torch_pin`, which is used to coordinate landing PRs in
`pytorch/pytorch` and `pytorch/ltc`.

To test a `pytorch/ltc` PR against a `pytorch/pytorch` PR or branch,
put the PR number or branch name in this file.
Example:

```
#32451
# or
my_awesome_branch # (must live in `pytorch/pytorch`)
```
