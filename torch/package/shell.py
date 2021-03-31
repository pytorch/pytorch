import cmd
import os
import sys
import networkx as nx
import matplotlib.pyplot as plt

from dep import Dep

class PackageShell(cmd.Cmd):
    intro = 'Welcome to the torch.package shell.   Type help or ? to list commands.\n'
    prompt = '(pkg-shell) '

    def __init__(self):
        super().__init__('tab', None, None)
        self.root = None
        self.externs = []

    def do_status(self, arg):
        'Check the current status of the package.  status'
        if not self.root:
            print("Run new_package before checking status")

        deps = Dep()

        for ext in self.externs:
            deps.extern(ext)

        deps.require_module(self.root)

        if not deps.broken_modules:
            print("All good!")

        if deps.broken_modules:
            print("Could not get source for the following modules:")
            for m in deps.broken_modules:
                print(m)

            print("Consider marking them as extern.")

        print()

    def do_extern(self, arg):
        'Mark a module as extern.  extern <module_name>'
        self.externs.append(arg)

    def do_show_graph(self, arg):
        'Show the dependency graph for the current package.  show_graph'
        deps = Dep()

        for ext in self.externs:
            deps.extern(ext)

        deps.require_module(self.root)

        nodes = set()
        for a, b in deps.debug_deps:
            nodes.add(a)
            nodes.add(b)

        G = nx.DiGraph()
        node_colours = []
        for n in nodes:
            G.add_node(n)
            if n == self.root:
                node_colours.append("aliceblue")
            elif n in deps.broken_modules:
                node_colours.append("red")
            else:
                node_colours.append("aquamarine")

        for t in deps.debug_deps:
            G.add_edge(*t)

        layout = nx.drawing.layout.shell_layout(G)
        nx.draw_networkx_nodes(G, layout, node_color=node_colours)
        nx.draw_networkx_edges(G, layout)
        nx.draw_networkx_labels(G, layout, verticalalignment="center_baseline")
        plt.savefig("path.png")

    def do_new_package(self, arg):
        'Start a new packaging exercise.  new_package <module_name>'
        self.root = arg

    def do_export(self, arg):
        'Export the current packaging settings to a file.  export <file_name>'
        with open(arg, "w") as f:
            f.write("with torch.package.PackageExporter(path) as e:\n")
            for ext in self.externs:
                f.write(f"\te.extern(\"{ext}\")\n")

            f.write(f"\te.save_pickle(\"{self.root}\", \"{self.root}\")\n")


if __name__ == "__main__":
    sys.path.append(os.getcwd())
    PackageShell().cmdloop()
