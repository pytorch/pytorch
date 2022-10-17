import argparse
from common import run, topics, get_features
from collections import defaultdict
import os
from pathlib import Path
import csv
import pprint
from common import get_commit_data_cache, features_to_dict
import re
import dataclasses
from typing import List


"""
Example Usages

Create a new commitlist for consumption by categorize.py.
Said commitlist contains commits between v1.5.0 and f5bc91f851.

    python commitlist.py --create_new tags/v1.5.0 f5bc91f851

Update the existing commitlist to commit bfcb687b9c.

    python commitlist.py --update_to bfcb687b9c

"""
@dataclasses.dataclass(frozen=True)
class Commit:
    commit_hash: str
    category: str
    topic: str
    title: str
    pr_link: str
    author: str

    # This is not a list so that it is easier to put in a spreadsheet
    accepter_1: str
    accepter_2: str
    accepter_3: str

    merge_into: str = None

    def __repr__(self):
        return f'Commit({self.commit_hash}, {self.category}, {self.topic}, {self.title})'

commit_fields = tuple(f.name for f in dataclasses.fields(Commit))

class CommitList:
    # NB: Private ctor. Use `from_existing` or `create_new`.
    def __init__(self, path: str, commits: List[Commit]):
        self.path = path
        self.commits = commits

    @staticmethod
    def from_existing(path):
        commits = CommitList.read_from_disk(path)
        return CommitList(path, commits)

    @staticmethod
    def create_new(path, base_version, new_version):
        if os.path.exists(path):
            raise ValueError('Attempted to create a new commitlist but one exists already!')
        commits = CommitList.get_commits_between(base_version, new_version)
        return CommitList(path, commits)

    @staticmethod
    def read_from_disk(path) -> List[Commit]:
        with open(path) as csvfile:
            reader = csv.DictReader(csvfile)
            rows = []
            for row in reader:
                if row.get("new_title", "") != "":
                    row["title"] = row["new_title"]
                filtered_rows = {k: row.get(k, "") for k in commit_fields}
                rows.append(Commit(**filtered_rows))
        return rows

    def write_result(self):
        self.write_to_disk_static(self.path, self.commits)

    @staticmethod
    def write_to_disk_static(path, commit_list):
        os.makedirs(Path(path).parent, exist_ok=True)
        with open(path, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(commit_fields)
            for commit in commit_list:
                writer.writerow(dataclasses.astuple(commit))

    def keywordInFile(file, keywords):
        for key in keywords:
            if key in file:
                return True
        return False

    @staticmethod
    def gen_commit(commit_hash):
        feature_item = get_commit_data_cache().get(commit_hash)
        features = features_to_dict(feature_item)
        category, topic = CommitList.categorize(features)
        a1, a2, a3 = (features["accepters"] + ("", "", ""))[:3]
        if features["pr_number"] is not None:
            pr_link = f"https://github.com/pytorch/pytorch/pull/{features['pr_number']}"
        else:
            pr_link = None

        return Commit(commit_hash, category, topic, features["title"], pr_link, features["author"], a1, a2, a3)

    @staticmethod
    def categorize(features):
        title = features['title']
        labels = features['labels']
        category = 'Uncategorized'
        topic = 'Untopiced'


        # We ask contributors to label their PR's appropriately
        # when they're first landed.
        # Check if the labels are there first.
        already_categorized = already_topiced = False
        for label in labels:
            if label.startswith('release notes: '):
                category = label.split('release notes: ', 1)[1]
                already_categorized = True
            if label.startswith('topic: '):
                topic = label.split('topic: ', 1)[1]
                already_topiced = True
        if already_categorized and already_topiced:
            return category, topic

        # update this to check if each file starts with caffe2
        if 'caffe2' in title:
            return 'caffe2', topic
        if '[codemod]' in title.lower():
            return 'skip', topic
        if 'Reverted' in labels:
            return 'skip', topic
        if 'bc_breaking' in labels:
            topic = 'bc-breaking'
        if 'module: deprecation' in labels:
            topic = 'deprecation'

        files_changed = features['files_changed']
        for file in files_changed:
            file_lowercase = file.lower()
            if CommitList.keywordInFile(file, ['docker/', '.circleci', '.github', '.jenkins', '.azure_pipelines']):
                category = 'releng'
                break
            # datapipe(s), torch/utils/data, test_{dataloader, datapipe}
            if CommitList.keywordInFile(file, ['torch/utils/data', 'test_dataloader', 'test_datapipe']):
                category = 'dataloader_frontend'
                break
            if CommitList.keywordInFile(file, ['torch/csrc/api', 'test/cpp/api']):
                category = 'cpp_frontend'
                break
            if CommitList.keywordInFile(file, ['distributed', 'c10d']):
                category = 'distributed'
                break
            if ('vulkan' in file_lowercase):
                category = 'vulkan'
                break
            if ('Foreach' in file_lowercase):
                category = 'foreach_frontend'
                break
            if 'onnx' in file_lowercase:
                category = 'onnx'
                break
            if CommitList.keywordInFile(file, ['torch/fx', 'test_fx']):
                category = 'fx'
                break
            if CommitList.keywordInFile(file, ['torch/ao', 'test/ao']):
                category = 'ao'
                break
            # torch/quantization, test/quantization, aten/src/ATen/native/quantized, torch/nn/{quantized, quantizable}
            if CommitList.keywordInFile(file, ['torch/quantization', 'test/quantization', 'aten/src/ATen/native/quantized', 'torch/nn/quantiz']):
                category = 'quantization'
                break
            if CommitList.keywordInFile(file, ['torch/package', 'test/package']):
                category = 'package'
                break
            if CommitList.keywordInFile(file, ['torch/csrc/jit/mobile', 'aten/src/ATen/native/metal', 'test/mobile', 'torch/backends/_nnapi/', 'test/test_nnapi.py']):
                category = 'mobile'
                break
            if CommitList.keywordInFile(file, ['aten/src/ATen/native/LinearAlgebra.cpp', 'test/test_linalg.py', 'torch/linalg']):
                category = 'linalg_frontend'
                break
            if CommitList.keywordInFile(file, ['torch/sparse', 'aten/src/ATen/native/sparse', 'torch/_masked/__init__.py']):
                category = 'sparse_frontend'
                break
            if CommitList.keywordInFile(file, ['tools/autograd']):
                category = 'autograd_frontend'
                break
            if CommitList.keywordInFile(file, ['test/test_nn.py', 'test/test_module.py', 'torch/nn/modules', 'torch/nn/functional.py']):
                category = 'nn_frontend'
                break
            if CommitList.keywordInFile(file, ['torch/csrc/jit', 'torch/jit']):
                category = 'jit'
                break
        else:
            # Below are some extra quick checks that aren't necessarily file-path related,
            # but I found that to catch a decent number of extra commits.
            if len(files_changed) > 0 and all([f_name.endswith('.cu') or f_name.endswith('.cuh') for f_name in files_changed]):
                category = 'cuda'
            elif '[PyTorch Edge]' in title:
                category = 'mobile'
            elif len(files_changed) == 1 and 'torch/testing/_internal/common_methods_invocations.py' in files_changed[0]:
                # when this is the only file changed, it's almost always an OpInfo change.
                category = 'python_frontend'
            elif len(files_changed) == 1 and 'torch/_torch_docs.py' in files_changed[0]:
                # individual torch_docs changes are usually for python ops
                category = 'python_frontend'


        return category, topic

    @staticmethod
    def get_commits_between(base_version, new_version):
        cmd = f'git merge-base {base_version} {new_version}'
        rc, merge_base, _ = run(cmd)
        assert rc == 0

        # Returns a list of something like
        # b33e38ec47 Allow a higher-precision step type for Vec256::arange (#34555)
        cmd = f'git log --reverse --oneline {merge_base}..{new_version}'
        rc, commits, _ = run(cmd)
        assert rc == 0

        log_lines = commits.split('\n')
        hashes, titles = zip(*[log_line.split(' ', 1) for log_line in log_lines])
        return [CommitList.gen_commit(commit_hash) for commit_hash in hashes]

    def filter(self, *, category=None, topic=None):
        commits = self.commits
        if category is not None:
            commits = [commit for commit in commits if commit.category == category]
        if topic is not None:
            commits = [commit for commit in commits if commit.topic == topic]
        return commits

    def update_to(self, new_version):
        last_hash = self.commits[-1].commit_hash
        new_commits = CommitList.get_commits_between(last_hash, new_version)
        self.commits += new_commits

    def stat(self):
        counts = defaultdict(lambda: defaultdict(int))
        for commit in self.commits:
            counts[commit.category][commit.topic] += 1
        return counts


def create_new(path, base_version, new_version):
    commits = CommitList.create_new(path, base_version, new_version)
    commits.write_result()

def update_existing(path, new_version):
    commits = CommitList.from_existing(path)
    commits.update_to(new_version)
    commits.write_result()

def rerun_with_new_filters(path):
    current_commits = CommitList.from_existing(path)
    for i in range(len(current_commits.commits)):
        c = current_commits.commits[i]
        if 'Uncategorized' in str(c):
            feature_item = get_commit_data_cache().get(c.commit_hash)
            features = features_to_dict(feature_item)
            category, topic = CommitList.categorize(features)
            current_commits[i] = dataclasses.replace(c, category=category, topic=topic)
    current_commits.write_result()

def get_hash_or_pr_url(commit: Commit):
    # cdc = get_commit_data_cache()
    pr_link = commit.pr_link
    if pr_link is None:
        return commit.commit_hash
    else:
        regex = r'https://github.com/pytorch/pytorch/pull/([0-9]+)'
        matches = re.findall(regex, pr_link)
        if len(matches) == 0:
            return commit.commit_hash

        return f'[#{matches[0]}]({pr_link})'

def to_markdown(commit_list: CommitList, category):
    def cleanup_title(commit):
        match = re.match(r'(.*) \(#\d+\)', commit.title)
        if match is None:
            return commit.title
        return match.group(1)

    merge_mapping = defaultdict(list)
    for commit in commit_list.commits:
        if commit.merge_into:
            merge_mapping[commit.merge_into].append(commit)

    cdc = get_commit_data_cache()
    lines = [f'\n## {category}\n']
    for topic in topics:
        lines.append(f'### {topic}\n')
        commits = commit_list.filter(category=category, topic=topic)
        if '_' in topic:
            commits.extend(commit_list.filter(category=category, topic=topic.replace('_', ' ')))
        if ' ' in topic:
            commits.extend(commit_list.filter(category=category, topic=topic.replace(' ', '_')))
        for commit in commits:
            if commit.merge_into:
                continue
            all_related_commits = merge_mapping[commit.commit_hash] + [commit]
            commit_list_md = ", ".join(get_hash_or_pr_url(c) for c in all_related_commits)
            result = f'- {cleanup_title(commit)} ({commit_list_md})\n'
            lines.append(result)
    return lines

def get_markdown_header(category):
    header = f"""
# Release Notes worksheet {category}

The main goal of this process is to rephrase all the commit messages below to make them clear and easy to read by the end user. You should follow the following instructions to do so:

* **Please cleanup, and format commit titles to be readable by the general pytorch user.** [Detailed intructions here](https://fb.quip.com/OCRoAbEvrRD9#HdaACARZZvo)
* Please sort commits into the following categories (you should not rename the categories!), I tried to pre-sort these to ease your work, feel free to move commits around if the current categorization is not good.
* Please drop any commits that are not user-facing.
* If anything is from another domain, leave it in the UNTOPICED section at the end and I'll come and take care of it.

The categories below are as follows:

* BC breaking: All commits that are BC-breaking. These are the most important commits. If any pre-sorted commit is actually BC-breaking, do move it to this section. Each commit should contain a paragraph explaining the rational behind the change as well as an example for how to update user code (guidelines here: https://quip.com/OCRoAbEvrRD9)
* Deprecations: All commits introducing deprecation. Each commit should include a small example explaining what should be done to update user code.
* new_features: All commits introducing a new feature (new functions, new submodule, new supported platform etc)
* improvements: All commits providing improvements to existing feature should be here (new backend for a function, new argument, better numerical stability)
* bug fixes: All commits that fix bugs and behaviors that do not match the documentation
* performance: All commits that are added mainly for performance (we separate this from improvements above to make it easier for users to look for it)
* documentation: All commits that add/update documentation
* Developers: All commits that are not end-user facing but still impact people that compile from source, develop into pytorch, extend pytorch, etc
"""

    return [header, ]


def main():
    parser = argparse.ArgumentParser(description='Tool to create a commit list')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--create_new', nargs=2)
    group.add_argument('--update_to')
    # I found this flag useful when experimenting with adding new auto-categorizing filters.
    # After running commitlist.py the first time, if you add any new filters in this file,
    # re-running with "rerun_with_new_filters" will update the existing commitlist.csv file,
    # but only affect the rows that were previously marked as "Uncategorized"
    group.add_argument('--rerun_with_new_filters', action='store_true')
    group.add_argument('--stat', action='store_true')
    group.add_argument('--export_markdown', action='store_true')
    group.add_argument('--export_csv_categories', action='store_true')
    parser.add_argument('--path', default='results/commitlist.csv')
    args = parser.parse_args()

    if args.create_new:
        create_new(args.path, args.create_new[0], args.create_new[1])
        return
    if args.update_to:
        update_existing(args.path, args.update_to)
        return
    if args.rerun_with_new_filters:
        rerun_with_new_filters(args.path)
        return
    if args.stat:
        commits = CommitList.from_existing(args.path)
        stats = commits.stat()
        pprint.pprint(stats)
        return

    if args.export_csv_categories:
        commits = CommitList.from_existing(args.path)
        categories = list(commits.stat().keys())
        for category in categories:
            print(f"Exporting {category}...")
            filename = f'results/export/result_{category}.csv'
            CommitList.write_to_disk_static(filename, commits.filter(category=category))
        return

    if args.export_markdown:
        commits = CommitList.from_existing(args.path)
        categories = list(commits.stat().keys())
        for category in categories:
            print(f"Exporting {category}...")
            lines = get_markdown_header(category)
            lines += to_markdown(commits, category)
            filename = f'results/export/result_{category}.md'
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'w') as f:
                f.writelines(lines)
        return
    raise AssertionError()

if __name__ == '__main__':
    main()
