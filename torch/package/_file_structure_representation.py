# -*- coding: utf-8 -*-
from typing import Dict, List

from ._glob_group import GlobPattern, _GlobGroup


class Folder:
    def __init__(self, name: str, is_dir: bool):
        self.name = name
        self.is_dir = is_dir
        self.children: Dict[str, Folder] = {}

    def get_folder(self, folders: List[str]):
        # builds path of folders if not yet built, returns last folder
        if len(folders) == 0:
            return self
        folder_name = folders[0]
        if folder_name not in self.children:
            self.children[folder_name] = Folder(folder_name, True)
        return self.children[folder_name].get_folder(folders[1:])

    def add_file(self, file_path):
        *folders, file = file_path.split("/")
        folder = self.get_folder(folders)
        folder.children[file] = Folder(file, False)

    def __str__(self):
        str_list: List[str] = []
        self.stringify_tree(str_list)
        return "".join(str_list)

    def stringify_tree(
        self, str_list: List[str], preamble: str = "", folder_ptr: str = "─── "
    ):
        space = "    "
        branch = "│   "
        tee = "├── "
        last = "└── "

        # add this folder's representation
        str_list.append(f"{preamble}{folder_ptr}{self.name}\n")

        # add folder's children representations
        if folder_ptr == tee:
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


def _create_folder_from_file_list(
    filename: str,
    file_list: List[str],
    include: "GlobPattern" = "**",
    exclude: "GlobPattern" = (),
) -> Folder:
    glob_pattern = _GlobGroup(include, exclude, "/")

    top_folder = Folder(filename, True)
    for file in file_list:
        if glob_pattern.matches(file):
            top_folder.add_file(file)
    return top_folder
