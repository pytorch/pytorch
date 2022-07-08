from collections import namedtuple
from pathlib import Path
import locale
import subprocess
import re
import requests
import os
import json

categories = [
    'Uncategorized',
    'distributed',
    'lazy',
    'hub',
    'mobile',
    'jit',
    'visualization',
    'onnx',
    'caffe2',
    'quantization',
    'amd',
    'rocm',
    'cuda',
    'cudnn',
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
    'composability',
    'meta_frontend',
    'nn_frontend',
    'linalg_frontend',
    'cpp_frontend',
    'python_frontend',
    'complex_frontend',
    'vmap_frontend',
    'autograd_frontend',
    'build_frontend',
    'memory_format_frontend',
    'foreach_frontend',
    'dataloader_frontend',
    'sparse_frontend'
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
    "not user facing",
    "security",
]


Features = namedtuple('Features', [
    'title',
    'body',
    'pr_number',
    'files_changed',
    'labels',
    'author',
    'accepters'
])


def dict_to_features(dct):
    return Features(
        title=dct['title'],
        body=dct['body'],
        pr_number=dct['pr_number'],
        files_changed=dct['files_changed'],
        labels=dct['labels'],
        author=dct['author'],
        accepters=tuple(dct['accepters']))


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
    with open(Path('~/.ghstackrc').expanduser(), 'r+') as f:
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


def github_data(pr_number):
    query = """
    {
      repository(owner: "pytorch", name: "pytorch") {
        pullRequest(number: %s ) {
          author {
            login
          }
          reviews(last: 5, states: APPROVED) {
            nodes {
              author {
                login
              }
            }
          }
          labels(first: 10) {
            edges {
              node {
                name
              }
            }
          }
        }
      }
    }
    """ % pr_number
    query = run_query(query)

    edges = query['data']['repository']['pullRequest']['labels']['edges']
    labels = [edge['node']['name'] for edge in edges]
    author = query['data']['repository']['pullRequest']['author']['login']
    nodes = query['data']['repository']['pullRequest']['reviews']['nodes']

    # using set to dedup multiple accepts from same accepter
    accepters = {node["author"]["login"] for node in nodes}
    accepters = tuple(sorted(accepters))

    return labels, author, accepters


def get_features(commit_hash):
    title, body, files_changed = (
        commit_title(commit_hash),
        commit_body(commit_hash),
        commit_files_changed(commit_hash))
    pr_number = parse_pr_number(body, commit_hash, title)
    labels = []
    author = ""
    accepters = tuple()
    if pr_number is not None:
        labels, author, accepters = github_data(pr_number)
    result = Features(title, body, pr_number, files_changed, labels, author, accepters)
    return result


_commit_data_cache = None

def get_commit_data_cache(path='results/data.json'):
    global _commit_data_cache
    if _commit_data_cache is None:
        _commit_data_cache = _CommitDataCache(path)
    return _commit_data_cache

class _CommitDataCache:
    def __init__(self, path):
        self.path = path
        self.data = {}
        if os.path.exists(path):
            self.data = self.read_from_disk()
        else:
            os.makedirs(Path(path).parent, exist_ok=True)

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
