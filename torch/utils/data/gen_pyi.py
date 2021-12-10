import os
import pathlib
from typing import Dict, List, Set, Tuple
from tools.codegen.gen import FileManager

def find_file_paths(dir_paths: List[str], files_to_exclude: Set[str]) -> Set[str]:
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


def parse_datapipe_file(file_path: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Given a path to file, parses the file and returns a dictionary of method names to function signatures.
    """
    method_to_signature, method_to_class_name = {}, {}
    with open(file_path) as f:
        open_paren_count = 0
        method_name, class_name, signature = "", "", ""
        skip = False
        for line in f.readlines():
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
    return method_to_signature, method_to_class_name


def parse_datapipe_files(file_paths: Set[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    methods_and_signatures, methods_and_class_names = {}, {}
    for path in file_paths:
        method_to_signature, method_to_class_name = parse_datapipe_file(path)
        methods_and_signatures.update(method_to_signature)
        methods_and_class_names.update(method_to_class_name)
    return methods_and_signatures, methods_and_class_names


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


def main() -> None:
    """
    .pyi generation for functional DataPipes Process
    # 1. Find files that we want to process (exclude the ones who don't)
    # 2. Parse method name and signature
    # 3. Remove first argument after self (unless it is "*datapipes"), default args, and spaces
    # 4. Inject file into template dataset.pyi.in
    TODO: The current implementation of this script only generates interfaces for built-in methods. To generate
          interface for user-defined DataPipes, consider changing `IterDataPipe.register_datapipe_as_function`.
    """

    files_to_exclude = {"__init__.py", "utils.py"}
    deprecated_files = {"httpreader.py", "linereader.py", "tararchivereader.py", "ziparchivereader.py"}

    os.chdir(str(pathlib.Path(__file__).parent.resolve()))
    iter_datapipes_file_path = "datapipes/iter"
    file_paths = find_file_paths([iter_datapipes_file_path], files_to_exclude=files_to_exclude.union(deprecated_files))
    methods_and_signatures, methods_and_class_names = parse_datapipe_files(file_paths)

    method_definitions = []
    for method_name, signature in methods_and_signatures.items():
        class_name = methods_and_class_names[method_name]
        method_definitions.append(f"# Functional form of '{class_name}'\ndef {method_name}({signature}): ...")
    method_definitions.sort(key=lambda s: s.split('\n')[1])  # sorting based on method_name

    fm = FileManager(install_dir='.', template_dir='.', dry_run=False)
    fm.write_with_template(filename="dataset.pyi",
                           template_fn="dataset.pyi.in",
                           env_callable=lambda: {'IterableDataPipeMethods': method_definitions})

if __name__ == '__main__':
    main()  # TODO: Run this script automatically within the build and CI process
