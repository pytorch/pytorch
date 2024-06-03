# Summary
These are a collection of scripts for access lists of commits between releases. There are other scripts for automatically generating labels for commits.

The release_notes Runbook and other supporting docs can be found here: [Release Notes Supporting Docs](https://drive.google.com/drive/folders/1J0Uwz8oE7TrdcP95zc-id1gdSBPnMKOR?usp=sharing)

An example of generated docs for submodule owners: [2.0 release notes submodule docs](https://drive.google.com/drive/folders/1zQtmF_ak7BkpGEM58YgJfnpNXTnFl25q?usp=share_link)

### Authentication:
First run the `test_release_notes.py` script to make sure you have the correct authentication set up. This script will try to access the GitHub API and will fail if you are not authenticated.

- If you have enabled ghstack then authentication should be set up correctly.
- Otherwise go to `https://github.com/settings/tokens` and create a token. You can either follow the steps to setup ghstack or set the env variable `GITHUB_TOKEN`.


## Steps:

### Part 1: getting a list of commits

You are going to get a list of commits since the last release in csv format. The usage is the following:
Assuming tags/v1.13.1 is last released version
From this directory run:
`python commitlist.py --create_new tags/v1.13.1 <commit_hash> `

This saves a commit list to `results/commitlist.csv`.  Please confirm visually that the oldest commits weren’t included in the branch cut for the last release as a sanity check.

NB: the commit list contains commits from the merge-base of tags/<most_recent_release_tag> and whatever commit hash you give it, so it may have commits that were cherry-picked to <most_recent_release_tag>!

* Go through the list of cherry-picked commits to the last release and delete them from results/commitlist.csv.
* This is done manually:
    * Look for all the PRs that were merged in the release branch with a github query like: https://github.com/pytorch/pytorch/pulls?q=is%3Apr+base%3Arelease%2F<most_recent_release_tag>+is%3Amerged
    *  Look at the commit history https://github.com/pytorch/pytorch/commits/release/<most_recent_release_tag>, to find all the direct push in the release branch (usually for reverts)


If you already have a commit list and want to update it, use the following command. This command can be helpful if there are cherry-picks to the release branch or if you’re categorizing commits throughout the three months up to a release. Warning: this is not very well tested. Make sure that you’re on the same branch (e.g., release/<upcoming_release_tag>) as the last time you ran this command, and that you always *commit* your csv before running this command to avoid losing work.

`python commitlist.py --update_to <commit_hash>`

### Part 2: categorizing commits

#### Exploration and cleanup

In this folder is an ipython notebook that I used for exploration and finding relevant commits. For example the commitlist attempts to categorize commits based off the `release notes:` label. Users of PyTorch often add new release notes labels. This Notebook has a cell that can help you identify new labels.

There is a list of all known categories defined in `common.py`. It has designations for types of categories as well such as `_frontend`.

The `categorize` function in commitlist.py does an adequate job of adding the appropriate categories. Since new categories though may be created for your release you may find it helpful to add new heuristics around files changed to help with categorization.

If you update the automatic categorization you can run the following to update the commit list.
`python commitlist.py --rerun_with_new_filters` Note that this will only update the commits in the commit list that have a category of "Uncategorized".

One you have dug through the commits and done as much automated categorization you can run the following for an interface to categorize any remaining commits.

#### Training a commit classifier
I added scripts to train a commit classifier from the set of labeled commits in commitlist.csv. This will utilize the title, author, and files changed features of the commits. The file requires torchtext, and tqdm. I had to install torchtext from source but if you are also a PyTorch developer this would likely already be installed.

- There should already exist a `results/` directory from gathering the commitlist.csv. The next step is to create `mkdir results/classifier`
- Run `python classifier.py --train` This will train the model and save for inference.
- Run `python categorize.py --use_classifier` This will pre-populate the output with the most likely category. And pressing enter will confirm selection.
 - Or run `python categorize.py` to label without the classifier.

The interface modifies results/commitlist.csv. If you want to take a coffee break, you can CTRL-C out of it (results/commitlist.csv gets written to on each categorization) and then commit and push results/commitlist.csv to a branch for safekeeping.

If you want to revert a change you just made, you can edit results/commitlist.csv directly.

For each commit, after choosing the category, you can also choose a topic. For the frontend category, you should take the time to do it to save time in the next step. For other categories, you can do it but only of you are 100% sure as it is confusing for submodule owners otherwise.

The categories are as follow:
 Be sure to update this list if you add a new category to common.py

* jit: Everything related to the jit (including tensorexpr)
* quantization: Everything related to the quantization mode/passes/operators
* mobile: Everything related to the mobile build/ops/features
* onnx: Everything related to onnx
* caffe2: Everything that happens in the caffe2 folder. No need to add any topics here as these are ignored (they don’t make it into the final release notes)
* distributed: Everything related to distributed training and rpc
* visualization: Everything related to tensorboard and visualization in general
* releng: Everything related to release engineering (circle CI, docker images, etc)
* amd: Everything related to rocm and amd CPUs
* cuda: Everything related to cuda backend
* benchmark: Everything related to the opbench folder and utils.benchmark submodule
* package: Everything related to torch.package
* performance as a product: All changes that improve perfs
* profiler: Everything related to the profiler
* composability: Everything related to the dispatcher and ATen native binding
* fx: Everything related to torch.fx
* code_coverage: Everything related to the code coverage tool
* vulkan: Everything related to vulkan support (mobile GPU backend)
* skip: Everything that is not end user or dev facing like code refactoring or internal implementation changes
* frontend: To ease your future work, we split things here (may be merged in the final document)
    * python_api
    * cpp_api
    * complex
    * vmap
    * autograd
    * build
    * memory_format
    * foreach
    * dataloader
    * nestedtensor
    * sparse
    * mps


The topics are as follow:

* bc_breaking: All commits marked as BC-breaking (the script should highlight them). If any other commit look like it could be BC-breaking, add it here as well!
* deprecation: All commits introducing deprecation. Should be clear from commit msg.
* new_features: All commits introducing a new feature (new functions, new submodule, new supported platform etc)
* improvements: All commits providing improvements to existing feature should be here (new backend for a function, new argument, better numerical stability)
* bug fixes: All commits that fix bugs and behaviors that do not match the documentation
* performance: All commits that are here mainly for performance (we separate this from improvements above to make it easier for users to look for it)
* documentation: All commits that add/update documentation
* devs: All commits that are not end-user facing but still impact people that compile from source, develop into pytorch, extend pytorch, cpp extensions, etc
* unknown


### Part 3: export categories to markdown

`python commitlist.py --export_markdown`

The above exports results/commitlist.csv to markdown by listing every commit under its respective category.
It will create one file per category in the results/export/ folder.

This part is a little tedious but it seems to work. May want to explore using pandoc to convert the markdown to google doc format.

1. Make sure you are using the light theme of VSCode.
2. Open a preview of the markdown file and copy the Preview.
3. In the correct google doc copy the preview and make sure to paste WITH formatting.
4. You can now send these google docs to the relevant submodule owners for review.
5. Install the google doc extension [docs to markdown](https://github.com/evbacher/gd2md-html)
6. Start to compile back down these markdown files into a single markdown file.

`TODO`: This is by far the most manual process and is ripe for automation. If the next person up would like to investigate Google Doc APIS there is some room hor improvement here.

### Part 4: Cherry Picks

You will likely have started this process prior to the branch-cut being finalized. This means Cherry Picks.
This was my process for keeping track. I use a notes app to log my progress as I periodically incorporate the new cherry picks.
I will have initially ran something like:
``` Bash
python commitlist.py --create_new tags/v1.13.1 <commit-hash>
```
I keep track of that commit-hash. Once there are some cherry-picks that you would like to incorporate I rebase the release branch to upstream
and run:
```Bash
python commitlist.py --update_to <latest-cherry-pick-hash>
```
I then run
``` Python
import pandas as pd

commit_list_df = pd.read_csv("results/commitlist.csv")
last_known_good_hash = "<the most recent hash>"

previous_index = commit_list_df[commit_list_df.commit_hash == last_known_good_hash].index.values[0]
cherry_pick_df = commit_list_df.iloc[previous_index+1:]
path = "<your_path>/cherry_picks.csv"
cherry_pick_df.to_csv(path, index=False)


from commitlist import CommitList, to_markdown
cherry_pick_commit_list = CommitList.from_existing(path)

import os
categories = list(cherry_pick_commit_list.stat().keys())
for category in categories:
    print(f"Exporting {category}...")
    lines =to_markdown(cherry_pick_commit_list, category)
    filename = f'/tmp/cherry_pick/results/result_{category}.md'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        f.writelines(lines)

```

This will create new markdown files only from cherry picked commits. And I manually copied and pasted these into the submodule google docs and commented so that
the submodule owners will see these new commits.


### Part 5: Pulling on the submodules into one
I pretty much followed the run book here. One thing I did was use the [markdown-all-in-one](https://marketplace.visualstudio.com/items?itemName=yzhang.markdown-all-in-one)
extension to create a table of contents which was really helpful in jumping to sections and copy and pasting the appropriate commits.

You will then create a release at [Pytorch Release](https://github.com/pytorch/pytorch/releases) and if you save as a draft you can see how it will be rendered.



#### Tidbits
You will probably have a release note that doesn't fit into the character limit of github. I used the following regex:
`\[#(\d+)\]\(https://github.com/pytorch/pytorch/pull/\d+\)` to replace the full lunks to (#<pull-request-number>).
This will get formatted correctly in the github UI and can be checked when creating a draft release.


The following markdown code is helpful for creating side-by-side tables of BC breaking/ deprecated code:


``` Markdown
<table>
<tr>
<th>PRIOR RELEASE NUM</th>
<th>NEW RELEASE NUM</th>
</tr>
<tr>
<td>

```Python
# Code Snippet 1
```

</td>
<td>

```Python
# Code Snippet 2
```

</td>
</tr>
</table>
```
