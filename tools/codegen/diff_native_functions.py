import argparse
import os
import requests  # TODO(jwtan): Figure out the conventional API to use within the PyTorch repository.
import subprocess
import sys
import tempfile

# TODO(jwtan): Figure out why we need this.
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root)

from tools.codegen.gen import parse_native_yaml

# Returns the current_version and the base_version if specified by users.
def get_versions() -> (str, str):
    # TODO(jwtan): It returns a lot of information including release notes. Figure out a way to only query the tag names.
    # TODO(jwtan): per_page=100 is not future proof.
    response = requests.get("https://api.github.com/repos/pytorch/pytorch/releases?per_page=100")
    releases = [release['tag_name'] for release in response.json()]

    current_version = releases[0]

    parser = argparse.ArgumentParser(description='Diff native functions. Please run this script in the open-source repository.')
    parser.add_argument(
        '-c',
        '--current-version',
        help='current PyTorch version, default: ' + current_version,
        default=current_version)
    parser.add_argument(
        '-b',
        '--base-version',
        help='base PyTorch version for comparison, default to the previous version')
    options = parser.parse_args()

    current_version = options.current_version
    base_version = options.base_version

    release_map = {value: index for index, value in enumerate(releases)}
    if (current_version not in release_map.keys()) or ((base_version is not None) and (base_version not in release_map.keys())):
        # TODO(jwtan): Figure out the conventional way to log/print error.
        print('Input versions are invalid. Valid versions are {}.'.format(releases))

    if base_version is None:
        base_version = releases[release_map[current_version] + 1]

    if release_map[current_version] >= release_map[base_version]:
        # TODO(jwtan): Figure out the conventional way to log/print error.
        print('Base version should be older than the current version.')

    return current_version, base_version

# Returns the corresponding commits for current_version and the base_version.
# TODO(jwtan): Can we combine get_versions() and get_commits(...)?
def get_commits(current_version: str, base_version: str) -> (str, str):
    # TODO(jwtan): per_page=100 is not future proof.
    response = requests.get("https://api.github.com/repos/pytorch/pytorch/tags?per_page=100")

    current_commit_hash = ''
    base_commit_hash = ''
    for tag in response.json():
        if tag['name'] == current_version:
            current_commit_hash = tag['commit']['sha']
        if tag['name'] == base_version:
            base_commit_hash = tag['commit']['sha']

    return current_commit_hash, base_commit_hash

# Writes the native_functions.yaml of the provided commit to a temp file, and returns the absolute path to the file.
# Caller should unlink the files when done.
def get_yaml(commit_hash: str):
    handle, path = tempfile.mkstemp()

    subprocess.run(['git', 'show', commit_hash + ':aten/src/ATen/native/native_functions.yaml'], stdout=handle)

    os.close(handle)
    return path

# Returns the hash key for the given NativeFunction.
def calculate_key(function: 'NativeFunction') -> str:
    return str(function.func.name)

# Returns the hash value for the given NativeFunction.
def calculate_value(function: 'NativeFunction') -> str:
    return str(function.func)

# TODO(jwtan): Consider adding support for the ToT native_functions.yaml.
# TODO(jwtan): Consider adding support for deleted/deprecated native functions.
def main() -> None:
    current_version, base_version = get_versions()
    assert current_version is not None
    assert base_version is not None
    # print(current_version, base_version)

    current_commit_hash, base_commit_hash = get_commits(current_version, base_version)
    assert current_commit_hash is not None
    assert base_commit_hash is not None
    # print(current_commit_hash, base_commit_hash)

    current_temp_path = get_yaml(current_commit_hash)
    base_temp_path = get_yaml(base_commit_hash)
    assert current_temp_path is not None
    assert base_temp_path is not None
    # print(current_temp_path, base_temp_path)

    current_native_functions = parse_native_yaml(current_temp_path)
    base_native_functions = parse_native_yaml(base_temp_path)
    os.unlink(current_temp_path)
    os.unlink(base_temp_path)
    # print(len(current_native_functions.native_functions))
    # print(len(base_native_functions.native_functions))

    # TODO(jwtan): Maybe we want the whole object of function.func instead of a serialized format.
    tags = {calculate_key(function): calculate_value(function) for function in base_native_functions.native_functions}
    # print(tags.keys())

    mismatched_jit_schema_functions = []
    new_functions = []
    for function in current_native_functions.native_functions:
        key = calculate_key(function)
        value = calculate_value(function)

        if key in tags.keys():
            if value != tags[key]:
                mismatched_jit_schema_functions.append((value, tags[key]))
            continue

        new_functions.append(value)
    mismatched_jit_schema_functions.sort()
    new_functions.sort()

    print('The following functions, total {}, have breaking changes:\n'.format(len(mismatched_jit_schema_functions)))
    for functions in mismatched_jit_schema_functions:
        print(current_version + ': ' + functions[0])
        print(base_version + ': ' + functions[1] + '\n')

    print('\n============================================\n')

    print('The following functions, total {}, are newly added in {}:\n'.format(len(new_functions), current_version))
    for function in new_functions:
        print(function)

if __name__ == '__main__':
    main()
