import argparse
import os
import textwrap
from common import categories, topics, CommitDataCache
from commitlist import CommitList

class Categorizer:
    def __init__(self, path, category='Uncategorized'):
        self.cache = CommitDataCache()
        self.commits = CommitList.from_existing(path)

        # Special categories: 'Uncategorized'
        # All other categories must be real
        self.category = category

    def categorize(self):
        commits = self.commits.filter(self.category)
        i = 0
        while i < len(commits):
            cur_commit = commits[i]
            next_commit = commits[i + 1] if i + 1 < len(commits) else None
            jump_to = self.handle_commit(cur_commit, i + 1, len(commits), commits)

            # Increment counter
            if jump_to is not None:
                i = jump_to
            elif next_commit is None:
                i = len(commits)
            else:
                i = commits.index(next_commit)

    def features(self, commit):
        return self.cache.get(commit.commit_hash)

    def potential_reverts_of(self, commit, commits):
        if 'Updating submodules' in commit.title:
            return []
        index = commits.index(commit)
        # -8 to remove the (#35011)
        cleaned_title = commit.title[:-10]
        # NB: the index + 2 is sketch
        return {(index + 2 + delta): cand for delta, cand in enumerate(commits[index + 1:])
                if cleaned_title in cand.title and
                commit.commit_hash != cand.commit_hash}

    def handle_commit(self, commit, i, total, commits):
        potential_reverts = self.potential_reverts_of(commit, commits)
        if potential_reverts:
            potential_reverts = f'!!!POTENTIAL REVERTS!!!: {potential_reverts}'
        else:
            potential_reverts = ""

        features = self.features(commit)

        breaking_alarm = ""
        if 'topic: bc-breaking' in features.labels:
            breaking_alarm += "!!!!!! BC BREAKING !!!!!!"

        if 'module: deprecation' in features.labels:
            breaking_alarm += "!!!!!! DEPRECATION !!!!!!"

        os.system('clear')
        view = textwrap.dedent(f'''\
[{i}/{total}]
================================================================================
{features.title}

{features.body}

Files changed: {features.files_changed}

Labels: {features.labels}

{potential_reverts} {breaking_alarm}

Current category: {commit.category}

Select from: {', '.join(categories)}

        ''')
        print(view)
        cat_choice = None
        while cat_choice is None:
            value = input('category> ').strip()
            if len(value) == 0:
                cat_choice = commit.category
                continue
            choices = [cat for cat in categories
                       if cat.startswith(value)]
            if len(choices) != 1:
                print(f'Possible matches: {choices}, try again')
                continue
            cat_choice = choices[0]
        print(f'\nSelected: {cat_choice}')
        print(f'\nCurrent topic: {commit.topic}')
        print(f'''Select from: {', '.join(topics)}''')
        topic_choice = None
        while topic_choice is None:
            value = input('topic> ').strip()
            if len(value) == 0:
                topic_choice = commit.topic
                continue
            choices = [cat for cat in topics
                       if cat.startswith(value)]
            if len(choices) != 1:
                print(f'Possible matches: {choices}, try again')
                continue
            topic_choice = choices[0]
        print(f'\nSelected: {topic_choice}')
        self.update_commit(commit, cat_choice, topic_choice)
        return None

    def update_commit(self, commit, category, topic):
        assert category in categories
        assert topic in topics
        commit.category = category
        commit.topic = topic
        self.commits.write_to_disk()

def main():
    parser = argparse.ArgumentParser(description='Tool to help categorize commits')
    parser.add_argument('--category', type=str, default='Uncategorized',
                        help='Which category to filter by. "Uncategorized", None, or a category name')
    parser.add_argument('--file', help='The location of the commits CSV',
                        default='results/commitlist.csv')

    args = parser.parse_args()
    categorizer = Categorizer(args.file, args.category)
    categorizer.categorize()


if __name__ == '__main__':
    main()
