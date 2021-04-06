import argparse
from common import run, topics
from collections import defaultdict
import os
import csv
import pprint
from common import CommitDataCache
import re


"""
Example Usages

Create a new commitlist for consumption by categorize.py.
Said commitlist contains commits between v1.5.0 and f5bc91f851.

    python commitlist.py --create_new tags/v1.5.0 f5bc91f851

Update the existing commitlist to commit bfcb687b9c.

    python commitlist.py --update_to bfcb687b9c

"""

class Commit:
    def __init__(self, commit_hash, category, topic, title):
        self.commit_hash = commit_hash
        self.category = category
        self.topic = topic
        self.title = title

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.commit_hash == other.commit_hash and \
            self.category == other.category and \
            self.topic == other.topic and \
            self.title == other.title

    def __repr__(self):
        return f'Commit({self.commit_hash}, {self.category}, {self.topic}, {self.title})'

class CommitList:
    # NB: Private ctor. Use `from_existing` or `create_new`.
    def __init__(self, path, commits):
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
    def read_from_disk(path):
        with open(path) as csvfile:
            reader = csv.reader(csvfile)
            rows = list(row for row in reader)
        assert all(len(row) >= 4 for row in rows)
        return [Commit(*row[:4]) for row in rows]

    def write_to_disk(self):
        path = self.path
        rows = self.commits
        with open(path, 'w') as csvfile:
            writer = csv.writer(csvfile)
            for commit in rows:
                writer.writerow([commit.commit_hash, commit.category, commit.topic, commit.title])

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
        return [Commit(commit_hash, 'Uncategorized', 'Untopiced', title) for commit_hash, title in zip(hashes, titles)]

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
    commits.write_to_disk()

def update_existing(path, new_version):
    commits = CommitList.from_existing(path)
    commits.update_to(new_version)
    commits.write_to_disk()

def to_markdown(commit_list, category):
    def cleanup_title(commit):
        match = re.match(r'(.*) \(#\d+\)', commit.title)
        if match is None:
            return commit.title
        return match.group(1)

    cdc = CommitDataCache()
    lines = [f'\n## {category}\n']
    for topic in topics:
        lines.append(f'### {topic}\n')
        commits = commit_list.filter(category=category, topic=topic)
        for commit in commits:
            result = cleanup_title(commit)
            maybe_pr_number = cdc.get(commit.commit_hash).pr_number
            if maybe_pr_number is None:
                result = f'- {result} ({commit.commit_hash})\n'
            else:
                result = f'- {result} ([#{maybe_pr_number}](https://github.com/pytorch/pytorch/pull/{maybe_pr_number}))\n'
            lines.append(result)
    return lines

def get_markdown_header(category):
    header = f"""
# Release Notes worksheet {category}

The main goal of this process is to rephrase all the commit messages below to make them clear and easy to read by the end user. You should follow the following instructions to do so:

* **Please cleanup, and format commit titles to be readable by the general pytorch user.** [Detailed intructions here](https://fb.quip.com/OCRoAbEvrRD9#HdaACARZZvo)
* Please sort commits into the following categories (you should not rename the categories!), I tried to pre-sort these to ease your work, feel free to move commits around if the current categorization is not good.
* Please drop any commits that are not user-facing.
* If anything is from another domain, leave it in the UNTOPICED section at the end and I’ll come and take care of it.

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

    return [header,]


def main():
    parser = argparse.ArgumentParser(description='Tool to create a commit list')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--create_new', nargs=2)
    group.add_argument('--update_to')
    group.add_argument('--stat', action='store_true')
    group.add_argument('--export_markdown', action='store_true')

    parser.add_argument('--path', default='results/commitlist.csv')
    args = parser.parse_args()

    if args.create_new:
        create_new(args.path, args.create_new[0], args.create_new[1])
        return
    if args.update_to:
        update_existing(args.path, args.update_to)
        return
    if args.stat:
        commits = CommitList.from_existing(args.path)
        stats = commits.stat()
        pprint.pprint(stats)
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
    assert False

if __name__ == '__main__':
    main()
