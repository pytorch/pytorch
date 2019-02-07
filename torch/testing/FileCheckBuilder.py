import torch

FileCheck = torch._C.FileCheck


# Handle to serialize series of checks from python
class FileCheckBuilder:
    def __init__(self):
        self.commands = []
        self.has_run = False

    def add_check(self, check, string):
        self.commands.append(check + string)
        return self

    # Checks that the string occurs, starting at the end of the most recent match
    def check(self, string):
        return self.add_check("CHECK:", string)

    # Checks that the string does not occur between the previous match and next match
    # Consecutive check_nots test against the same previous match and next match
    def check_not(self, string):
        return self.add_check("CHECK-NOT:", string)

    # Checks that the string occurs on the same line as the previous match
    def check_same(self, string):
        return self.add_check("CHECK-SAME:", string)

    # Checks that the string occurs on the line immediately following the previous match
    def check_next(self, string):
        return self.add_check("CHECK-NEXT:", string)

    # Checks that the string occurs count number of times
    def check_count(self, string, count):
        return self.add_check("CHECK-COUNT-" + str(count) + ":", string)

    # A series of consecutive check_dags get turned into a group of checks
    # which can appear in any order relative to each other.
    def check_dag(self, string):
        return self.add_check("CHECK-DAG:", string)

    def reset(self):
        self.commands = []

    def run(self, string):
        check_str = "\n".join(self.commands)
        self.has_run = True
        FileCheck.run(check_str, string)

    # Avoid accidentally not running FileCheck
    def __del__(self):
        if not self.has_run:
            print("You have not run this instance of FileCheckBuilder"
                  ". If this was intentional set builder.has_run = True.")
