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
Right now we have two branches:
* `main` -- follows the `main` branch of the upstream
* `poolside-main` -- contains our changes and should be rebased over `main`

We commit our changes to `poolside-main`.
If we want to commit our changes to the upstream, we branch from `main` and cherry-pick the corresponding commits from `poolside-main`.

The changes we made should be summarized in the [poolside-changes.md](poolside-changes.md) file.

Feel free to contact Dmitrii Emelianenko or Vadim Markovtsev for any questions regarding this repo.

