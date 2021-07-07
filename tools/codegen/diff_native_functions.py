# TODO(jwtan): Prune some imports.
import argparse
import os
import requests # TODO(jwtan): Figure out the conventional API to use within the PyTorch repo.
import subprocess
import sys
import tempfile
import yaml

from collections import OrderedDict, defaultdict, namedtuple

# TODO(jwtan): Figure out why we need this.
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root)

from tools.codegen.gen import LineLoader, error_check_native_functions, parse_native_yaml
from tools.codegen.model import (Argument, DispatchKey, FunctionSchema,
                                 Location, NativeFunction,
                                 NativeFunctionsGroup, OperatorName,
                                 BackendIndex, BackendMetadata,
                                 OptionalType, SchemaKind, SelfArgument,
                                 TensorOptionsArguments, Type, Variant,
                                 assert_never, is_cuda_dispatch_key,
                                 is_generic_dispatch_key)
from tools.codegen.utils import Target, concatMap, context, mapMaybe, YamlDumper, YamlLoader

# Returns the current_version and the base_version if specified by users.
def get_versions() -> (str, str):
    # TODO(jwtan): It returns a lot of information including release notes. Figure out a way to only query the tag names.
    response = requests.get("https://api.github.com/repos/pytorch/pytorch/releases")
    releases = [release['tag_name'] for release in response.json()]

    current_version = releases[0]
    base_version = releases[1]

    parser = argparse.ArgumentParser(description='Diff native functions. Please run this script in the OSS repo.')
    parser.add_argument(
        '-c',
        '--current-version',
        help='current PyTorch version, default: ' + current_version,
        default=current_version)
    parser.add_argument(
        '-b',
        '--base-version',
        help='base PyTorch version for comparison, default: ' + base_version,
        default=base_version)
    options = parser.parse_args()

    current_version = options.current_version
    base_version = options.base_version

    release_map = {value: index for index, value in enumerate(releases)}
    if (current_version not in release_map.keys()) or (base_version not in release_map.keys()):
        # TODO(jwtan): Figure out the conventional way to log/print error.
        print('Input versions are invalid. Valid versions are {}.'.format(releases))
    if release_map[current_version] >= release_map[base_version]:
        # TODO(jwtan): Figure out the 131313conventional way to log/print error.
        print('Base version should be older than the current version.')

    return current_version, base_version

# Returns the corresponding commits for current_version and the base_version.
# TODO(jwtan): Can we combine get_versions() and get_commits(...)?
def get_commits(current_version: str, base_version: str) -> (str, str):
    response = requests.get("https://api.github.com/repos/pytorch/pytorch/tags")
    tags = {tag['name']: tag['commit']['sha'] for tag in response.json()}

    return tags[current_version], tags[base_version]

# Writes the native_functions.yaml of the provided commit to a temp file, and returns the absolute path to the file.
# Caller should unlink the files when done.
def get_yaml(commit_hash: str):
    handle, path = tempfile.mkstemp()

    subprocess.run(['git', 'show', commit_hash + ':aten/src/ATen/native/native_functions.yaml'], stdout=handle)

    os.close(handle)
    return path

# TODO(jwtan): Consider adding support for the ToT native_functions.yaml.
def main() -> None:
    current_version, base_version = get_versions()
    # print(current_version, base_version)

    current_commit_hash, base_commit_hash = get_commits(current_version, base_version)
    # print(current_commit_hash, base_commit_hash)

    current_temp_path = get_yaml(current_commit_hash)
    base_temp_path = get_yaml(base_commit_hash)
    # print(current_temp_path, base_temp_path)

    current_native_functions = parse_native_yaml(current_temp_path)
    base_native_functions = parse_native_yaml(base_temp_path)
    os.unlink(current_temp_path)
    os.unlink(base_temp_path)
    # print(len(current_native_functions.native_functions))
    # print(len(base_native_functions.native_functions))

    # TODO(jwtan): Maybe we want the whole object of function.func instead of a serialized format.
    tags = {(function.func.name.name.base, function.func.name.overload_name): str(function.func) for function in base_native_functions.native_functions}
    # print(tags)

    mismatched_jit_schema_functions = []
    new_functions = []
    for function in current_native_functions.native_functions:
        key = (function.func.name.name.base, function.func.name.overload_name)
        value = str(function.func)

        if key in tags.keys():
            if value != tags[key]:
                mismatched_jit_schema_functions.append(value)
            continue

        new_functions.append(value)

    print('The following funtions has breaking changes:\n')
    for function in mismatched_jit_schema_functions:
        print(function)

    print('\n============================================\n')

    print('The following funtions are newly added:\n')
    for function in new_functions:
        print(function)

if __name__ == '__main__':
    main()
