1. Make sure both Bert's repo and the official pytorch repo are added as remotes.

```
$ git remote -v
bert    git@github.com:bertmaher/pytorch.git (fetch)
bert    git@github.com:bertmaher/pytorch.git (push)
origin  git@github.com:pytorch/pytorch.git (fetch)
origin  git@github.com:pytorch/pytorch.git (push)
...
```
You might see https address instead of the ssh one (e.g. `https://github.com/pytorch/pytorch.git`), which should also be fine if you only plan to pull from it.

If you don't have these remotes, add the missing ones with
```
git remote add <name> <link>
```

E.g.
```
git remote add pt https://github.com/pytorch/pytorch.git
```

You can remove a remote if you need with
```
git remote remove <name>
```

2. Fetch all the remotes:
```
git fetch --all
```

3. Stash/commit all your local changes
```
git stash # OR
git commit -a -m "My local changes"
```

4. Checkout branch that you'd like to rebase on top of the master. Assuming we'd want to rebase the `pytorch_fusion` branch from Bert's repo, you could do:
```
git checkout pytorch_fusion          # Checkout local 'pytorch_fusion' branch
git reset --hard bert/pytorch_fusion # This will replace the current, 'pytorch_fusion', branch with the version from Bert's repo
```

5. Rebase your branch on top of the latest master branch:
```
git rebase origin/master
```
If you're lucky and there are not conflicts, you will end up with a rebased branch.
In the other case, manually resolve the conflicts: for every conflict, do:
 - `git status` to find "both modified" files - that's where the conflicts are
 - Manually edit these files to resolve the conflict.
 - Mark the conflict as resolved by adding these files with `git add FILENAME`
 - Once conflicts in all files are resolved, run `git rebase --continue`
 - At any point you can run `git rebase --abort` and you will escape to the state before the rebase step.

6. Push to our (Bert's repo). That will have to be a force-push, so make sure to:
 - Double check what you're going to push (e.g. with `git log`) - compare that the new branch and the old branch (`bert/pytorch_fusion`) have the same commits on top, the only difference is the last master commit in the branch.
 - Announce that you're going to force-push the main branch. Other people will have to rebase their changes after that.
 - Push with local branch 'pytorch_fusion' to the Bert's repo under the same name: `git push bert -f pytorch_fusion:pytorch_fusion`

7. ...

8. Profit!
