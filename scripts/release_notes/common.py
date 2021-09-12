from collections import namedtuple
from os.path import expanduser
import locale
import subprocess
import re
import requests
import os
import json

categories = [
    'Uncategorized',
    'distributed',
    'mobile',
    'jit',
    'visualization',
    'onnx',
    'caffe2',
    'quantization',
    'amd',
    'cuda',
    'benchmark',
    'profiler',
    'performance_as_product',
    'package',
    'dispatcher',
    'releng',
    'fx',
    'code_coverage',
    'vulkan',
    'skip',
    'cpp_frontend',
    'python_frontend',
    'complex_frontend',
    'vmap_frontend',
    'autograd_frontend',
    'build_frontend',
    'memory_format_frontend',
    'foreach_frontend',
    'dataloader_frontend'
]

topics = [
    'bc_breaking',
    'deprecations',
    'new_features',
    'improvements',
    'bug_fixes',
    'performance',
    'docs',
    'devs',
    'Untopiced',
]


Features = namedtuple('Features', [
    'title',
    'body',
    'pr_number',
    'files_changed',
    'labels',
])


def dict_to_features(dct):
    return Features(
        title=dct['title'],
        body=dct['body'],
        pr_number=dct['pr_number'],
        files_changed=dct['files_changed'],
        labels=dct['labels'])


def features_to_dict(features):
    return dict(features._asdict())


def run(command):
    """Returns (return-code, stdout, stderr)"""
    p = subprocess.Popen(command, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, shell=True)
    output, err = p.communicate()
    rc = p.returncode
    enc = locale.getpreferredencoding()
    output = output.decode(enc)
    err = err.decode(enc)
    return rc, output.strip(), err.strip()


def commit_body(commit_hash):
    cmd = f'git log -n 1 --pretty=format:%b {commit_hash}'
    ret, out, err = run(cmd)
    return out if ret == 0 else None


def commit_title(commit_hash):
    cmd = f'git log -n 1 --pretty=format:%s {commit_hash}'
    ret, out, err = run(cmd)
    return out if ret == 0 else None


def commit_files_changed(commit_hash):
    cmd = f'git diff-tree --no-commit-id --name-only -r {commit_hash}'
    ret, out, err = run(cmd)
    return out.split('\n') if ret == 0 else None


def parse_pr_number(body, commit_hash, title):
    regex = r'Pull Request resolved: https://github.com/pytorch/pytorch/pull/([0-9]+)'
    matches = re.findall(regex, body)
    if len(matches) == 0:
        if 'revert' not in title.lower() and 'updating submodules' not in title.lower():
            print(f'[{commit_hash}: {title}] Could not parse PR number, ignoring PR')
        return None
    if len(matches) > 1:
        print(f'[{commit_hash}: {title}] Got two PR numbers, using the first one')
        return matches[0]
    return matches[0]


def get_ghstack_token():
    pattern = 'github_oauth = (.*)'
    with open(expanduser('~/.ghstackrc'), 'r+') as f:
        config = f.read()
    matches = re.findall(pattern, config)
    if len(matches) == 0:
        raise RuntimeError("Can't find a github oauth token")
    return matches[0]

token = get_ghstack_token()
headers = {"Authorization": f"token {token}"}

def run_query(query):
    request = requests.post('https://api.github.com/graphql', json={'query': query}, headers=headers)
    if request.status_code == 200:
        return request.json()
    else:
        raise Exception("Query failed to run by returning code of {}. {}".format(request.status_code, query))


def gh_labels(pr_number):
    query = f"""
    {{
      repository(owner: "pytorch", name: "pytorch") {{
        pullRequest(number: {pr_number}) {{
          labels(first: 10) {{
            edges {{
              node {{
                name
              }}
            }}
          }}
        }}
      }}
    }}
    """
    query = run_query(query)
    edges = query['data']['repository']['pullRequest']['labels']['edges']
    return [edge['node']['name'] for edge in edges]


def get_features(commit_hash, return_dict=False):
    title, body, files_changed = (
        commit_title(commit_hash),
        commit_body(commit_hash),
        commit_files_changed(commit_hash))
    pr_number = parse_pr_number(body, commit_hash, title)
    labels = []
    if pr_number is not None:
        labels = gh_labels(pr_number)
    result = Features(title, body, pr_number, files_changed, labels)
    if return_dict:
        return features_to_dict(result)
    return result

class CommitDataCache:
    def __init__(self, path='results/data.json'):
        self.path = path
        self.data = {}
        if os.path.exists(path):
            self.data = self.read_from_disk()

    def get(self, commit):
        if commit not in self.data.keys():
            # Fetch and cache the data
            self.data[commit] = get_features(commit)
            self.write_to_disk()
        return self.data[commit]

    def read_from_disk(self):
        with open(self.path, 'r') as f:
            data = json.load(f)
            data = {commit: dict_to_features(dct)
                    for commit, dct in data.items()}
        return data

    def write_to_disk(self):
        data = {commit: features._asdict() for commit, features in self.data.items()}
        with open(self.path, 'w') as f:
            json.dump(data, f)
