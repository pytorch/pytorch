import json
from typing import List, Tuple
from dependency_graph_template import TEMPLATE
import pdb


def visualize_dependencies(dependencies: List[Tuple[str, str]], filename: str):
    nodes = set()
    edges = set()

    for src, target in dependencies:
        src_id = src.replace(".", "_")
        target_id = target.replace(".", "_")
        nodes.add(src_id)
        nodes.add(target_id)
        edges.add(f"{src_id}___{target_id}")

        parent_name = ""

        for atom in src.split("."):
            parent_name = parent_name + ("" if parent_name == "" else "_")  + atom
            nodes.add(parent_name)

        parent_name = ""

        for atom in target.split("."):
            parent_name = parent_name + ("" if parent_name == "" else "_")  + atom
            nodes.add(parent_name)

    elements = []

    for node in nodes:
        node_dict = {
            "data": {
                "id": node,
            },
            "group": "nodes",
            "removed": False,
            "selected": False,
            "selectable": True,
            "locked": False,
            "grabbable": True,
            "classes": "",
        }

        if "_" in node:
            node_dict["data"]["label"] = node[node.rfind("_")+1:]
            node_dict["data"]["parent"] = node[:node.rfind("_")]
        else:
            node_dict["data"]["label"] = node

        elements.append(node_dict)

    for edge in edges:
        src, target = edge.split("___")

        edge_dict = {
            "data": {
                "id": edge,
                "source": src,
                "target": target,
                "arrow": "triangle-backcurve",
            },
            "group": "edges",
            "removed": False,
            "selected": False,
            "selectable": True,
            "locked": False,
            "grabbable": True,
            "classes": "",
        }

        elements.append(edge_dict)

    graph = TEMPLATE.replace("_________ELEMENTS_GO_HERE_________", json.dumps(elements))

    with open(filename, "w") as f:
        f.write(graph)



if __name__ == "__main__":
    deps = [
        ("mycode", "module.a"),
        ("mycode", "module.b"),
        ("mycode", "othermodule"),
        ("other", "module.a"),
    ]

    visualize_dependencies(deps, "/tmp/graph.html")
