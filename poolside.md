# poolside's fork of pytorch

Why this fork?
* We might need to fix problems with pytorch that block or slowdown training of our models.
* We don't want to wait until such problems are fixed in the upstream.
* We need custom build configurations anyway.

With that said, we don't want to maintain a completely separate version of pytorch.

To avoid this, we try to follow these principles:
* Updating to the current upstream should not be problematic and we should do it often.
* If we commit any changes:
  * We should minimize potential breaking changes/conflicts with the upstream.
  * If the risk for potential conflicts is significant, we should consider commiting the changes to the upstream.

## Repository organisation:
Right now we have three branches:
* `main` -- A periodically updated snapshot of the `main` branch of the upstream;
* `poolside-cherry-picks` -- contains specific cherry-picked commits in-between updates of the `main` branch. Should be rebased over `main`;
* `poolside-main` -- contains our changes and should be rebased over `poolside-cherry-picks`.

We commit our changes to `poolside-main`.
If we want to commit our changes to the upstream, we branch from `main` and cherry-pick the corresponding commits from `poolside-main`.

The changes we made should be summarized in the [poolside-changes.md](poolside-changes.md) file.

Feel free to contact Dmitrii Emelianenko or Vadim Markovtsev for any questions regarding this repo.

## Pulling upstream

Add the original repo as remote to fetch future changes. Make sure you also disable push on the remote (as you are not allowed to push to it anyway).

```shell
git remote add upstream https://github.com/pytorch/pytorch
git remote set-url --push upstream DISABLE
```

You can list all your remotes with `git remote -v`. You should see
```text
origin  git@github.com:poolsideai/pytorch.git (fetch)
origin  git@github.com:poolsideai/pytorch.git (push)
upstream        https://github.com/pytorch/pytorch (fetch)
upstream        DISABLE (push)
```

> When you push, do so on `origin` with `git push origin`.

When you want to pull changes from `upstream`, there are four steps: 

### Select which nightly you want to rebase onto
We are making our changes over the bleeding edge nightly releases.

Visit the nightly branch in the upstream repo: https://github.com/pytorch/pytorch/commits/nightly/ .

The commit messages for nightly builds contain commit hash from the main branch.
*Important*: you need the commit hash from the commit message, not the commit hash from the nightly branch.

Let's assume you have selected a nightly with date `$NIGHTLY_DATE` and the corresponding commit hash `$COMMIT`.
The following instructions will assume that `$COMMIT` is an ancestor of our `main` in the `upstream/main` (i.e. we are rebasing onto a newer version).

### Reset the `main` branch to `$COMMIT`.

```shell
git fetch upstream main
# we should have fetched the $COMMIT from main by now
git checkout main
git reset --hard $COMMIT
# This should work as `main` branch is an ancestor of $COMMIT originally
git push origin main
```

### Reset `poolside-cherry-picks` to `main`
We assume here that our cherry-picked commits are already present in `main` after the last step.
Otherwise you need to find the original commits and cherry-pick them on top of the local branch before the push.

```shell
git checkout poolside-cherry-picks
git reset --hard main
# cherry-pick the desired commits here if they are not in main already
...
git push origin poolside-cherry-picks --force-with-lease
```

### Rebase `poolside-main` onto `main`

```shell
git checkout poolside-main
git rebase poolside-cherry-picks
```

Now you need to fix the rebase conflicts if any.

You also need to Update the date in `PYTORCH_BUILD_VERSION_PREFIX` variable in our CI to `$NIGHTLY_DATE` [here](https://github.com/poolsideai/pytorch/blob/poolside-main/.github/workflows/poolside-nightly-build.yaml).
This is needed to represent that our fork is rebased over the specific pytorch nightly.

Now we're ready to push the updated version:

```shell
git push origin poolside-main --force-with-lease

```

# Publishing a new version to CodeArtifact:
Run the workflow defined here: https://github.com/poolsideai/pytorch/blob/poolside-main/.github/workflows/poolside-nightly-build.yaml
in the Actions tab. 

Note: this only works on `poolside-main` branch (no runners will be found otherwise).
