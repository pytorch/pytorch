import json
import locale
import os
import re
import subprocess
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path

import requests


@dataclass
class CategoryGroup:
    name: str
    categories: list


frontend_categories = [
    "meta",
    "nn",
    "linalg",
    "cpp",
    "python",
    "complex",
    "vmap",
    "autograd",
    "build",
    "memory_format",
    "foreach",
    "dataloader",
    "sparse",
    "nested tensor",
    "optimizer",
]

pytorch_2_categories = [
    "dynamo",
    "inductor",
]

# These will all get mapped to quantization
quantization = CategoryGroup(
    name="quantization",
    categories=[
        "quantization",
        "AO frontend",
        "AO Pruning",
    ],
)

# Distributed has a number of release note labels we want to map to one
distributed = CategoryGroup(
    name="distributed",
    categories=[
        "distributed",
        "distributed (c10d)",
        "distributed (composable)",
        "distributed (ddp)",
        "distributed (fsdp)",
        "distributed (rpc)",
        "distributed (sharded)",
    ],
)

categories = (
    [
        "Uncategorized",
        "lazy",
        "hub",
        "mobile",
        "jit",
        "visualization",
        "onnx",
        "caffe2",
        "amd",
        "rocm",
        "cuda",
        "cpu",
        "cudnn",
        "xla",
        "benchmark",
        "profiler",
        "performance_as_product",
        "package",
        "dispatcher",
        "releng",
        "fx",
        "code_coverage",
        "vulkan",
        "skip",
        "composability",
        # 2.0 release
        "mps",
        "intel",
        "functorch",
        "gnn",
        "distributions",
        "serialization",
    ]
    + [f"{category}_frontend" for category in frontend_categories]
    + pytorch_2_categories
    + [quantization.name]
    + [distributed.name]
)


topics = [
    "bc breaking",
    "deprecation",
    "new features",
    "improvements",
    "bug fixes",
    "performance",
    "docs",
    "devs",
    "Untopiced",
    "not user facing",
    "security",
]


Features = namedtuple(
    "Features",
    ["title", "body", "pr_number", "files_changed", "labels", "author", "accepters"],
)


def dict_to_features(dct):
    return Features(
        title=dct["title"],
        body=dct["body"],
        pr_number=dct["pr_number"],
        files_changed=dct["files_changed"],
        labels=dct["labels"],
        author=dct["author"],
        accepters=tuple(dct["accepters"]),
    )


def features_to_dict(features):
    return dict(features._asdict())


def run(command):
    """Returns (return-code, stdout, stderr)"""
    p = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )
    output, err = p.communicate()
    rc = p.returncode
    enc = locale.getpreferredencoding()
    output = output.decode(enc)
    err = err.decode(enc)
    return rc, output.strip(), err.strip()


def commit_body(commit_hash):
    cmd = f"git log -n 1 --pretty=format:%b {commit_hash}"
    ret, out, err = run(cmd)
    return out if ret == 0 else None


def commit_title(commit_hash):
    cmd = f"git log -n 1 --pretty=format:%s {commit_hash}"
    ret, out, err = run(cmd)
    return out if ret == 0 else None


def commit_files_changed(commit_hash):
    cmd = f"git diff-tree --no-commit-id --name-only -r {commit_hash}"
    ret, out, err = run(cmd)
    return out.split("\n") if ret == 0 else None


def parse_pr_number(body, commit_hash, title):
    regex = r"Pull Request resolved: https://github.com/pytorch/pytorch/pull/([0-9]+)"
    matches = re.findall(regex, body)
    if len(matches) == 0:
        if "revert" not in title.lower() and "updating submodules" not in title.lower():
            print(f"[{commit_hash}: {title}] Could not parse PR number, ignoring PR")
        return None
    if len(matches) > 1:
        print(f"[{commit_hash}: {title}] Got two PR numbers, using the first one")
        return matches[0]
    return matches[0]


def get_ghstack_token():
    pattern = "github_oauth = (.*)"
    with open(Path("~/.ghstackrc").expanduser(), "r+") as f:
        config = f.read()
    matches = re.findall(pattern, config)
    if len(matches) == 0:
        raise RuntimeError("Can't find a github oauth token")
    return matches[0]


def get_token():
    env_token = os.environ.get("GITHUB_TOKEN")
    if env_token is not None:
        print("using GITHUB_TOKEN from environment variable")
        return env_token
    else:
        return get_ghstack_token()


token = get_token()

headers = {"Authorization": f"token {token}"}


def run_query(query):
    request = requests.post(
        "https://api.github.com/graphql", json={"query": query}, headers=headers
    )
    if request.status_code == 200:
        return request.json()
    else:
        raise Exception(  # noqa: TRY002
            f"Query failed to run by returning code of {request.status_code}. {request.json()}"
        )


_ERRORS = []
_MAX_ERROR_LEN = 20


def github_data(pr_number):
    query = (
        """
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
    """
        % pr_number
    )
    query = run_query(query)
    if query.get("errors"):
        global _ERRORS
        _ERRORS.append(query.get("errors"))
        if len(_ERRORS) < _MAX_ERROR_LEN:
            return [], "None", ()
        else:
            raise Exception(  # noqa: TRY002
                f"Got {_MAX_ERROR_LEN} errors: {_ERRORS}, please check if"
                " there is something wrong"
            )
    edges = query["data"]["repository"]["pullRequest"]["labels"]["edges"]
    labels = [edge["node"]["name"] for edge in edges]
    author = query["data"]["repository"]["pullRequest"]["author"]["login"]
    nodes = query["data"]["repository"]["pullRequest"]["reviews"]["nodes"]

    # using set to dedup multiple accepts from same accepter
    accepters = {node["author"]["login"] for node in nodes}
    accepters = tuple(sorted(accepters))

    return labels, author, accepters


def get_features(commit_hash):
    title, body, files_changed = (
        commit_title(commit_hash),
        commit_body(commit_hash),
        commit_files_changed(commit_hash),
    )
    pr_number = parse_pr_number(body, commit_hash, title)
    labels = []
    author = ""
    accepters = tuple()
    if pr_number is not None:
        labels, author, accepters = github_data(pr_number)
    result = Features(title, body, pr_number, files_changed, labels, author, accepters)
    return result


_commit_data_cache = None


def get_commit_data_cache(path="results/data.json"):
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
        with open(self.path) as f:
            data = json.load(f)
            data = {commit: dict_to_features(dct) for commit, dct in data.items()}
        return data

    def write_to_disk(self):
        data = {commit: features._asdict() for commit, features in self.data.items()}
        with open(self.path, "w") as f:
            json.dump(data, f)
