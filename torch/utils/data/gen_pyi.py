import json
import os
import pathlib
import requests
from typing import Dict, List, Optional, Set, Tuple
from tools.codegen.gen import FileManager


def find_file_paths(dir_paths, files_to_exclude: Set[str]) -> Set[str]:
    """
    When given a path to a directory, returns the paths to the relevant files within it.
    This function does NOT recursive traverse to subdirectories.
    """
    paths: Set[str] = set()
    for dir_path in dir_paths:
        all_files = os.listdir(dir_path)
        python_files = {fname for fname in all_files if ".py" == fname[-3:]}
        filter_files = {fname for fname in python_files if fname not in files_to_exclude}
        paths.update({os.path.join(dir_path, fname) for fname in filter_files})
    return paths


def extract_method_name(line: str) -> str:
    """
    Extracts method name from decorator in the form of "@functional_datapipe({method_name})"
    """
    if "(\"" in line:
        start_token, end_token = "(\"", "\")"
    elif "(\'" in line:
        start_token, end_token = "(\'", "\')"
    else:
        raise RuntimeError(f"Unable to find appropriate method name within line:\n{line}")
    start, end = line.find(start_token) + len(start_token), line.find(end_token)
    return line[start:end]


def extract_class_name(line: str) -> str:
    """
    Extracts class name from class definition in the form of "class {CLASS_NAME}({Type}):"
    """
    start_token = "class "
    end_token = "("
    start, end = line.find(start_token) + len(start_token), line.find(end_token)
    return line[start:end]


def parse_datapipe_file(file_path: str = "", file_content: str = "") -> Tuple[Dict[str, str], Dict[str, str], Set[str]]:
    """
    Given a path to file, parses the file and returns a dictionary of method names to function signatures.
    """
    if file_path != "" and file_content != "":
        raise RuntimeError("Only one of 'file_path' or 'lines' should be given in parse_datapipe_file")
    method_to_signature, method_to_class_name, special_output_type = {}, {}, set()
    if file_path != "":
        f = open(file_path)
        lines = f.readlines()
    else:
        lines = file_content.splitlines()
    open_paren_count = 0
    method_name, class_name, signature = "", "", ""
    skip = False
    for line in lines:
        if line.count("\"\"\"") % 2 == 1:
            skip = not skip
        if skip or "\"\"\"" in line:  # Skipping comment/example blocks
            continue
        if "@functional_datapipe" in line:
            method_name = extract_method_name(line)
            continue
        if method_name and "class " in line:
            class_name = extract_class_name(line)
            continue
        if method_name and ("def __init__(" in line or "def __new__(" in line):
            if "def __new__(" in line:
                special_output_type.add(method_name)
            open_paren_count += 1
            start = line.find("(") + len("(")
            line = line[start:]
        if open_paren_count > 0:
            open_paren_count += line.count('(')
            open_paren_count -= line.count(')')
            if open_paren_count == 0:
                end = line.rfind(')')
                signature += line[:end]
                method_to_signature[method_name] = process_signature(signature)
                method_to_class_name[method_name] = class_name
                method_name, class_name, signature = "", "", ""
            elif open_paren_count < 0:
                raise RuntimeError("open parenthesis count < 0. This shouldn't be possible.")
            else:
                signature += line.strip('\n').strip(' ')
    if file_path != "":
        f.close()
    return method_to_signature, method_to_class_name, special_output_type


def parse_datapipe_files(file_paths: Optional[Set[str]] = None, file_contents: Optional[List[str]] = None)\
        -> Tuple[Dict[str, str], Dict[str, str], Set[str]]:

    if file_paths is not None and file_contents is not None:
        raise RuntimeError("Only one of 'file_paths' or 'file_contents' should be given in parse_datapipe_file")

    methods_and_signatures, methods_and_class_names, methods_with_special_output_types = {}, {}, set()
    if file_paths is not None and file_contents is None:
        iterable = iter(file_paths)
    else:
        iterable = iter(file_contents)

    for item in iterable:
        if file_paths is not None and file_contents is None:
            method_to_signature, method_to_class_name, methods_needing_special_output_types =\
                parse_datapipe_file(file_path=item)
        else:
            method_to_signature, method_to_class_name, methods_needing_special_output_types = \
                parse_datapipe_file(file_content=item)
        methods_and_signatures.update(method_to_signature)
        methods_and_class_names.update(method_to_class_name)
        methods_with_special_output_types.update(methods_needing_special_output_types)
    return methods_and_signatures, methods_and_class_names, methods_with_special_output_types


def split_outside_bracket(line: str, delimiter: str = ",") -> List[str]:
    """
    Given a line of text, split it on comma unless the comma is within a bracket '[]'.
    """
    bracket_count = 0
    curr_token = ""
    res = []
    for char in line:
        if char == "[":
            bracket_count += 1
        elif char == "]":
            bracket_count -= 1
        elif char == delimiter and bracket_count == 0:
            res.append(curr_token)
            curr_token = ""
            continue
        curr_token += char
    res.append(curr_token)
    return res


def process_signature(line: str) -> str:
    """
    Given a raw function signature, clean it up by removing the self-referential datapipe argument,
    default arguments of input functions, newlines, and spaces.
    """
    tokens: List[str] = split_outside_bracket(line)
    for i, token in enumerate(tokens):
        tokens[i] = token.strip(' ')
        if token == "cls":
            tokens[i] = "self"
        elif i > 0 and ("self" == tokens[i - 1]) and (tokens[i][0] != "*"):
            # Remove the datapipe after 'self' or 'cls' unless it has '*'
            tokens[i] = ""
        elif "Callable =" in token:  # Remove default argument if it is a function
            head, default_arg = token.rsplit("=", 2)
            tokens[i] = head.strip(' ') + "= ..."
    tokens = [t for t in tokens if t != ""]
    line = ', '.join(tokens)
    return line


_TORCH_DATA_ONLY_TEXT = " - This DataPipe is only available through the 'torchdata' library."


def get_method_definitions(root_path: str,
                           files_to_exclude: Set[str],
                           deprecated_files: Set[str],
                           default_output_type: str,
                           method_to_special_output_type: Dict[str, str],
                           file_contents: Optional[List[str]] = None,
                           from_torchdata: bool = False) -> List[str]:
    """
    .pyi generation for functional DataPipes Process
    # 1. Find files that we want to process (exclude the ones who don't)
    # 2. Parse method name and signature
    # 3. Remove first argument after self (unless it is "*datapipes"), default args, and spaces
    """
    if root_path != "" and file_contents is None:
        os.chdir(str(pathlib.Path(__file__).parent.resolve()))
        file_paths = find_file_paths(dir_paths=[root_path],
                                     files_to_exclude=files_to_exclude.union(deprecated_files))
        methods_and_signatures, methods_and_class_names, methods_w_special_output_types =\
            parse_datapipe_files(file_paths=file_paths)
    elif file_contents is not None:
        methods_and_signatures, methods_and_class_names, methods_w_special_output_types = \
            parse_datapipe_files(file_contents=file_contents)

    method_definitions = []
    for method_name, arguments in methods_and_signatures.items():
        class_name = methods_and_class_names[method_name]
        if method_name in methods_w_special_output_types:
            output_type = method_to_special_output_type[method_name]
        else:
            output_type = default_output_type
        additional_text = _TORCH_DATA_ONLY_TEXT if from_torchdata else ""
        method_definitions.append(f"# Functional form of '{class_name}'{additional_text}\n"
                                  f"def {method_name}({arguments}) -> {output_type}: ...")
    method_definitions.sort(key=lambda s: s.split('\n')[1])  # sorting based on method_name
    return method_definitions


_TORCHDATA_GITHUB_REPO_TREE = "https://api.github.com/repos/pytorch/data/git/trees/main?recursive=1"
_TORCHDATA_RAW_URL = "https://raw.githubusercontent.com/pytorch/data/main/"


def is_dp_file(desired_path: str, input_path: str):
    return desired_path in input_path and ".py" in input_path


def get_torch_data_files(dir_path: str):
    res = requests.get(_TORCHDATA_GITHUB_REPO_TREE)
    json_res = json.loads(res.content.decode('utf-8'))
    files = json_res['tree']
    relevant_files = [item for item in files if is_dp_file(dir_path, item['path'])]
    raw_files = []
    for f in relevant_files:
        file_res = requests.get(f"{_TORCHDATA_RAW_URL}{f['path']}")
        raw_files.append(file_res.text)
    return raw_files


def main() -> None:
    """
    # Inject file into template dataset.pyi.in
    TODO: The current implementation of this script only generates interfaces for built-in methods. To generate
          interface for user-defined DataPipes, consider changing `IterDataPipe.register_datapipe_as_function`.
    """

    iterDP_file_path: str = "datapipes/iter"
    iterDP_files_to_exclude: Set[str] = {"__init__.py", "utils.py"}
    iterDP_deprecated_files: Set[str] = {"httpreader.py", "linereader.py", "tararchivereader.py", "ziparchivereader.py"}
    iterDP_method_to_special_output_type: Dict[str, str] = {"demux": "List[IterDataPipe]", "fork": "List[IterDataPipe]"}

    iter_method_definitions = get_method_definitions(root_path=iterDP_file_path,
                                                     files_to_exclude=iterDP_files_to_exclude,
                                                     deprecated_files=iterDP_deprecated_files,
                                                     default_output_type="IterDataPipe",
                                                     method_to_special_output_type=iterDP_method_to_special_output_type)

    torchdata_iterDP_dir_path = "torchdata/datapipes/iter/"
    file_contents = get_torch_data_files(torchdata_iterDP_dir_path)
    torchdata_iterDP_to_exclude = {"__init__.py"}
    torchdata_iterDP_method_to_special_output_type: Dict[str, str] = {"end_caching": "IterDataPipe"}
    torchdata_iterDP_definitions = get_method_definitions(root_path="",
                                                          file_contents=file_contents,
                                                          files_to_exclude=torchdata_iterDP_to_exclude,
                                                          deprecated_files=set(),
                                                          default_output_type="IterDataPipe",
                                                          method_to_special_output_type=torchdata_iterDP_method_to_special_output_type,
                                                          from_torchdata=True)

    iter_method_definitions += torchdata_iterDP_definitions

    # TODO: When functional DataPipe is called, raise an error to warn users that they need to import TorchData

    mapDP_file_path: str = "datapipes/map"
    mapDP_files_to_exclude: Set[str] = {"__init__.py", "utils.py"}
    mapDP_deprecated_files: Set[str] = set()
    mapDP_method_to_special_output_type: Dict[str, str] = {}

    map_method_definitions = get_method_definitions(root_path=mapDP_file_path,
                                                    files_to_exclude=mapDP_files_to_exclude,
                                                    deprecated_files=mapDP_deprecated_files,
                                                    default_output_type="MapDataPipe",
                                                    method_to_special_output_type=mapDP_method_to_special_output_type)

    # TODO: Get the definitions from the TorchData Map DPs

    # TODO: Test if this causes an error if 'torchdata' isn't installed
    torch_data_import_statements = ["from torchdata.datapipes.iter.util.extractor import CompressionType"]
    torch_data_imports = ["try:"] +\
                         ["    " + s for s in torch_data_import_statements] +\
                         ["except ImportError:", "    pass"]

    fm = FileManager(install_dir='.', template_dir='.', dry_run=False)
    fm.write_with_template(filename="dataset.pyi",
                           template_fn="dataset.pyi.in",
                           env_callable=lambda: {'IterableDataPipeMethods': iter_method_definitions,
                                                 'MapDataPipeMethods': map_method_definitions,
                                                 'TorchDataImports': torch_data_imports})


if __name__ == '__main__':
    main()  # TODO: Run this script automatically within the build and CI process
