# -*- coding: utf-8 -*-
from typing import Dict, List

from .glob_group import GlobPattern, GlobGroup


class Directory:
    """A file structure representation. Organized as Directory nodes who have lists of
    their Directory children."""

    def __init__(self, name: str, is_dir: bool):
        self.name = name
        self.is_dir = is_dir
        self.children: Dict[str, Directory] = {}

    def get_dir(self, dirs: List[str]):
        """Builds path of Directories if not yet built. Return last directory 
        in list.

        Args:
            dirs (List[str]): List of directory names that are treated like a path.
        """
        # builds path of directories if not yet built, returns last directory
        if len(dirs) == 0:
            return self
        dir_name = dirs[0]
        if dir_name not in self.children:
            self.children[dir_name] = Directory(dir_name, True)
        return self.children[dir_name].get_dir(dirs[1:])

    def add_file(self, file_path: str):
        """Adds a file to a Directory. 

        Args:
            file_path (str): Path of file to add. Last element is added as a file while
                other paths items are added as directories. 
        """
        *dirs, file = file_path.split("/")
        dir = self.get_dir(dirs)
        dir.children[file] = Directory(file, False)

    def has_file(self, filename: str):
        """ Returns `True` if a Directory contains the specified file.

        Args:
            filename (str): Path of file to search for.
        """
        lineage = filename.split("/", maxsplit=1)
        child = lineage[0]
        grandchildren = lineage[1] if len(lineage) > 1 else None
        if child in self.children.keys():
            if grandchildren is None:
                return True
            else:
                return self.children[child].has_file(grandchildren)
        return False

    def __str__(self):
        str_list: List[str] = []
        self.stringify_tree(str_list)
        return "".join(str_list)

    def stringify_tree(
        self, str_list: List[str], preamble: str = "", dir_ptr: str = "─── "
    ):
        """Recursive method to generate print-friendly version of a Directory."""
        space = "    "
        branch = "│   "
        tee = "├── "
        last = "└── "

        # add this directory's representation
        str_list.append(f"{preamble}{dir_ptr}{self.name}\n")

        # add directory's children representations
        if dir_ptr == tee:
            preamble = preamble + branch
        else:
            preamble = preamble + space

        file_keys: List[str] = []
        dir_keys: List[str] = []
        for key, val in self.children.items():
            if val.is_dir:
                dir_keys.append(key)
            else:
                file_keys.append(key)

        for index, key in enumerate(sorted(dir_keys)):
            if (index == len(dir_keys) - 1) and len(file_keys) == 0:
                self.children[key].stringify_tree(str_list, preamble, last)
            else:
                self.children[key].stringify_tree(str_list, preamble, tee)
        for index, file in enumerate(sorted(file_keys)):
            pointer = last if (index == len(file_keys) - 1) else tee
            str_list.append(f"{preamble}{pointer}{file}\n")


def _create_directory_from_file_list(
    filename: str,
    file_list: List[str],
    include: "GlobPattern" = "**",
    exclude: "GlobPattern" = (),
) -> Directory:
    """ Return a Directory file structure representation created from a list of files.

    Args:
        filename (str): The name given to the top-level directory that will be the
            relative root for all file paths found in the file_list.

        file_list (List[str]): List of files to add to the top-level directory.

        include (Union[List[str], str]): An optional pattern that limits what is included from the file_list to
            files whose name matches the pattern. 

        exclude (Union[List[str], str]): An optional pattern that excludes files whose name match the pattern. 
    """
    glob_pattern = GlobGroup(include, exclude=exclude, separator="/")

    top_dir = Directory(filename, True)
    for file in file_list:
        if glob_pattern.matches(file):
            top_dir.add_file(file)
    return top_dir
