import argparse
import urllib.request
from typing import List


def output_should_run(should_run):
    if should_run:
        print("::set-output name=should_run::true")
    else:
        print("::set-output name=should_run::false")


def get_diff(pr_number: str) -> str:
    url = f"https://patch-diff.githubusercontent.com/raw/pytorch/pytorch/pull/{pr_number}.diff"
    with urllib.request.urlopen(url) as response:
        diff = response.read().decode()
    return diff


def get_files_from_diff(diff: str) -> List[str]:
    lines = diff.split("\n")
    files = set()
    for line in lines:
        # TODO: maybe use unidiff?
        line = line.strip()
        prefixes = {
            "--- a/",
            "--- b/",
            "+++ b/",
            "+++ a/",
        }
        for prefix in prefixes:
            if line.startswith(prefix):
                files.add(line.lstrip(prefix))

    return list(files)


def is_docs_pr(files: List[str]) -> bool:
    def is_doc_file(f):
        return f.endswith(".md") or f.endswith(".rst") or f.endswith(".txt")

    return all(is_doc_file(f) for f in files)


def determinate_for_job(pr: str, build_environment: str) -> bool:
    try:
        diff = get_diff(pr)
    except Exception:
        return True

    DOCS_BUILD = "linux-xenial-py3.6-gcc5.4"
    files = get_files_from_diff(diff)
    doc_match = is_docs_pr(files)
    build_match = build_environment == DOCS_BUILD
    print(f"Found files {files}")
    print(f"Is PR doc files only? {doc_match}")
    print(f"Does build match '{DOCS_BUILD}'? {build_match}")
    return doc_match and build_match


if __name__ == "__main__":
    # TODO: move this all into a lambda that can be updated async
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--build_environment", required=True, help="pr number to determine against"
    )
    parser.add_argument("--pr", required=True, help="pr number to determine against")
    args = parser.parse_args()

    output_should_run(determinate_for_job(args.pr, args.build_environment))
