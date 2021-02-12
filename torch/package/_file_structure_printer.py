from typing import List, Dict
import pathlib


def _print_file_structure(
    filename: str, records: List[str], include: str = "**", exclude: str = ""
):
    space = "    "
    branch = "│   "
    tee = "├── "
    last = "└── "
    start = "─── "

    class Folder:
        def __init__(self, name: str):
            self.name = name
            self.sub_folders: Dict[str, Folder] = {}
            self.sub_files: List[str] = []

        # builds path of folders if not yet built, returns last folder
        def get_folder(self, folders: List[str]):
            if len(folders) == 0:
                return self
            folder_name = folders[0]
            if folder_name not in self.sub_folders:
                self.sub_folders[folder_name] = Folder(folder_name)
            return self.sub_folders[folder_name].get_folder(folders[1:])

        def add_file(self, file_path):
            *folders, file = file_path.split("/")
            folder = self.get_folder(folders)
            folder.sub_files.append(file)

        def stringify_tree(
            self, str_list: List[str], extention: str = "", folder_ptr: str = start
        ):
            self.sub_files.sort()
            str_list.append(f"{extention}{folder_ptr}{self.name}\n")
            if folder_ptr == tee:
                extention = extention + branch
            else:
                extention = extention + space
            for index, key in enumerate(sorted(self.sub_folders)):
                if (index == len(self.sub_folders) - 1) and len(self.sub_files) == 0:
                    self.sub_folders[key].stringify_tree(str_list, extention, last)
                else:
                    self.sub_folders[key].stringify_tree(str_list, extention, tee)
            for index, file in enumerate(self.sub_files):
                pointer = last if (index == len(self.sub_files) - 1) else tee
                str_list.append(f"{extention}{pointer}{file}\n")

    top_folder = Folder(filename)
    for file in records:
        if pathlib.PurePath(file).match(include) and (
            exclude == "" or not pathlib.PurePath(file).match(exclude)
        ):
            top_folder.add_file(file)

    str_list: List[str] = []
    top_folder.stringify_tree(str_list)
    return "".join(str_list)
