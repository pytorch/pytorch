import io
import types
from typing import List, Union

import torch


class DependencyExplorer:
    """This class allows you to explore and understand a module or object's dependencies so that
    you can make more informed decisions about which packages to mark as extern and mock when
    using a PackageExporter.
    """

    # The module or object whose dependencies are being explored.
    root: Union[str, types.ModuleType, str]
    # Temporary buffer used by self.exporter.
    buffer: io.BytesIO
    # PackageExporter instance used to determine package-ability and dependencies.
    exporter: torch.package.PackageExporter

    def __init__(self, obj: Union[str, types.ModuleType, object]):
        self.root = obj
        self.buffer = io.BytesIO()
        # Create PackageExporter with raise_packaging_errors=False so that module import errors do not raise
        # and are instead recorded as "broken modules."
        self.exporter = torch.package.PackageExporter(self.buffer, verbose=False, raise_packaging_errors=False)

    def _refresh(self):
        """Reinvoke save_module/save_pickle on self.exporter in order to determine if
        any new externs/mocks have made the root module/object packageable.
        """
        # Reset exporter state.
        # TODO: Should this be an exporter method?
        self.buffer.seek(0)
        self.exporter.broken_modules.clear()
        self.exporter.extern_modules.clear()
        self.exporter.provided.clear()
        self.exporter.debug_deps.clear()

        if isinstance(self.root, (str, types.ModuleType)):
            self.exporter.save_module(self.root)
        else:
            self.exporter.save_pickle(self.root)

    def can_package(self) -> bool:
        """Check whether the root object can be successfully packaged with the current set of
        externs and mocks.
        """
        self._refresh()
        return not self.exporter.broken_modules

    def get_unresolved_dependencies(self) -> List[torch.package.BrokenModule]:
        """Get the list of broken modules encountered when trying to package the root object
        with the provided externs and mocks.
        """
        self._refresh()
        return list(self.exporter.broken_modules)

    def show_dependency_graph(self):
        """Show a visualization of the dependency graph of the root object.
        TODO: Replace this with a better visualization (networkx is an adequate
        placeholder for now).
        """
        # Import networkx.
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
        except ImportError:
            print("show_dependency_graph uses networkx to render the dependency graph;"
                  "install it with `pip install networks`")
            return

        # Refresh to get the updated dependency graph.
        self._refresh()

        # self.exporter.debug_deps contains edges; extract nodes from this list.
        nodes = set()
        for a, b in self.exporter.debug_deps:
            nodes.add(a)
            nodes.add(b)

        # This function determines whether a given module is broken or not.
        # It is used to assign nodes the right colour in the graph.
        def is_broken(n: str) -> bool:
            for broken, _ in self.exporter.broken_modules:
                if broken == n and not self.exporter._can_implicitly_extern(n):
                    return True

            return False

        # Create graph.
        G = nx.DiGraph()

        # Add nodes, with the right colours:
        #   root -> blue
        #   broken module -> red
        #   non-root packageable dependency -> aquamarine
        node_colours = []
        for n in nodes:
            G.add_node(n)
            if n == self.root:
                node_colours.append("aliceblue")
            elif is_broken(n):
                node_colours.append("red")
            else:
                node_colours.append("aquamarine")

        # Add edges.
        for edge in self.exporter.debug_deps:
            G.add_edge(*edge)

        # Layout and draw graph.
        layout = nx.drawing.layout.shell_layout(G)
        nx.draw_networkx_nodes(G, layout, node_color=node_colours)
        nx.draw_networkx_edges(G, layout)
        nx.draw_networkx_labels(G, layout, verticalalignment="center_baseline")

        # Show graph.
        plt.show()

    def extern(self, include, *, exclude=(), allow_empty=True):
        """See PackageExporter.extern."""
        self.exporter.extern(include, exclude=exclude, allow_empty=allow_empty)

    def mock(self, include, *, exclude=(), allow_empty=True):
        """See PackageExporter.mock."""
        self.exporter.mock(include, exclude=exclude, allow_empty=allow_empty)
