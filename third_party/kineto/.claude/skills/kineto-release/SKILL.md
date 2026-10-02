---
name: kineto-release
description: Update the third_party/kineto submodule in PyTorch to the latest commit from this kineto repo and commit the change. Use when updating the kineto submodule hash for a release.
disable-model-invocation: true
argument-hint: "[commit-hash]"
allowed-tools: Bash(git:*), Read, AskUserQuestion
---

# Update Kineto Submodule

Update the `third_party/kineto` submodule in a local PyTorch repo to point to a
commit from this Kineto repo.

If a commit hash is provided via `$ARGUMENTS`, use that as the target hash.
Otherwise, use the tip of main from this repo.

## Prerequisites

Before starting, you need the path to the local PyTorch repo. If you do not
already know it from prior conversation context, ask the user:

> "What is the path to your local PyTorch repo?"

The PyTorch repo is required — do not proceed without it. Store the path as
`$PYTORCH` for the steps below.

## Steps

1. **(PyTorch) Get the current submodule hash:**
   ```
   cd $PYTORCH && git submodule status third_party/kineto | awk '{print $1}' | sed 's/^[-+]//'
   ```

2. **(Kineto) Get the target hash.** If `$ARGUMENTS` is provided, use it. Otherwise, get
   the tip of main from this repo:
   ```
   git log -1 --format='%H' main
   ```
   Do NOT fetch or pull from remote -- you do not have permissions. The repo is
   already up to date.

3. **(PyTorch) Create a branch** named `release_YYYY_MM_DD_<short_new_hash>` (using today's
   date and the short target commit hash) and switch to it.

4. **(Kineto) List included commits** between old and new hash:
   ```
   git log --format='- %s %h' <old_hash>..<new_hash> | sed 's/(#/(pytorch\/kineto#/g'
   ```

5. **(PyTorch) Update the submodule** to the new hash. The submodule's local
   object store likely won't have the target commit, so fetch from the local
   Kineto repo first:
   ```
   cd $PYTORCH/third_party/kineto && git fetch <path_to_kineto_repo> main && git checkout <new_hash>
   ```

6. **(PyTorch) Commit** with a message in this format:
   ```
   Update third_party/kineto submodule to <short_new_hash>

   Includes the following commits:

   - <message1> <hash1>
   - <message2> <hash2>
   - ...

   ```

If the current hash already matches the target, inform the user that kineto is
already up to date and stop.
