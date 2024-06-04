# mypy: allow-untyped-defs

class GraphModuleOpUpgrader:

    def __init__(
            self,
            *args,
            **kwargs
    ):
        pass


    def upgrade(self, exported_program):
        return exported_program
