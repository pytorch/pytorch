"""Unit tests for explicit image comparison with pytest-mpl."""

import pytest

import networkx as nx

pytest.importorskip("pytest_mpl")

mpl = pytest.importorskip("matplotlib")
mpl.use("PS")
plt = pytest.importorskip("matplotlib.pyplot")
plt.rcParams["text.usetex"] = False
np = pytest.importorskip("numpy")


@pytest.mark.mpl_image_compare
def test_display_house_with_colors():
    """
    Originally, I wanted to use the exact samge image as test_house_with_colors.
    But I can't seem to find the correct value for the margins to get the figures
    to line up perfectly. To the human eye, these visualizations are basically the
    same.
    """
    G = nx.house_graph()
    fig, ax = plt.subplots()
    nx.set_node_attributes(
        G, {0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (1, 1), 4: (0.5, 2.0)}, "pos"
    )
    nx.set_node_attributes(
        G,
        {
            n: {
                "size": 3000 if n != 4 else 2000,
                "color": "tab:blue" if n != 4 else "tab:orange",
            }
            for n in G.nodes()
        },
    )
    nx.display(
        G,
        node_pos="pos",
        edge_alpha=0.5,
        edge_width=6,
        node_label=None,
        node_border_color="k",
    )
    ax.margins(0.17)
    plt.tight_layout()
    plt.axis("off")
    return fig


@pytest.mark.mpl_image_compare
def test_display_labels_and_colors():
    """See 'Labels and Colors' gallery example"""
    fig, ax = plt.subplots()
    G = nx.cubical_graph()
    pos = nx.spring_layout(G, seed=3113794652)  # positions for all nodes
    nx.set_node_attributes(G, pos, "pos")  # Will not be needed after PR 7571
    labels = iter(
        [
            r"$a$",
            r"$b$",
            r"$c$",
            r"$d$",
            r"$\alpha$",
            r"$\beta$",
            r"$\gamma$",
            r"$\delta$",
        ]
    )
    nx.set_node_attributes(
        G,
        {
            n: {
                "size": 800,
                "alpha": 0.9,
                "color": "tab:red" if n < 4 else "tab:blue",
                "label": {"label": next(labels), "size": 22, "color": "whitesmoke"},
            }
            for n in G.nodes()
        },
    )

    nx.display(G, node_pos="pos", edge_color="tab:grey")

    # The tricky bit is the highlighted colors for the edges
    edgelist = [(0, 1), (1, 2), (2, 3), (0, 3)]
    nx.set_edge_attributes(
        G,
        {
            (u, v): {
                "width": 8,
                "alpha": 0.5,
                "color": "tab:red",
                "visible": (u, v) in edgelist,
            }
            for u, v in G.edges()
        },
    )
    nx.display(G, node_pos="pos", node_visible=False)
    edgelist = [(4, 5), (5, 6), (6, 7), (4, 7)]
    nx.set_edge_attributes(
        G,
        {
            (u, v): {
                "color": "tab:blue",
                "visible": (u, v) in edgelist,
            }
            for u, v in G.edges()
        },
    )
    nx.display(G, node_pos="pos", node_visible=False)

    plt.tight_layout()
    plt.axis("off")
    return fig


@pytest.mark.mpl_image_compare
def test_display_complex():
    import itertools as it

    fig, ax = plt.subplots()
    G = nx.MultiDiGraph()
    nodes = "ABC"
    prod = list(it.product(nodes, repeat=2)) * 4
    G = nx.MultiDiGraph()
    for i, (u, v) in enumerate(prod):
        G.add_edge(u, v, w=round(i / 3, 2))
    nx.set_node_attributes(G, nx.spring_layout(G, seed=3113794652), "pos")
    csi = it.cycle([f"arc3,rad={r}" for r in it.accumulate([0.15] * 4)])
    nx.set_edge_attributes(G, {e: next(csi) for e in G.edges(keys=True)}, "curvature")
    nx.set_edge_attributes(
        G,
        {
            tuple(e): {"label": w, "bbox": {"alpha": 0}}
            for *e, w in G.edges(keys=True, data="w")
        },
        "label",
    )
    nx.apply_matplotlib_colors(G, "w", "color", mpl.colormaps["inferno"], nodes=False)
    nx.display(G, canvas=ax, node_pos="pos")

    plt.tight_layout()
    plt.axis("off")
    return fig


@pytest.mark.mpl_image_compare
def test_display_shortest_path():
    fig, ax = plt.subplots()
    G = nx.Graph()
    G.add_nodes_from(["A", "B", "C", "D", "E", "F", "G", "H"])
    G.add_edge("A", "B", weight=4)
    G.add_edge("A", "H", weight=8)
    G.add_edge("B", "C", weight=8)
    G.add_edge("B", "H", weight=11)
    G.add_edge("C", "D", weight=7)
    G.add_edge("C", "F", weight=4)
    G.add_edge("C", "I", weight=2)
    G.add_edge("D", "E", weight=9)
    G.add_edge("D", "F", weight=14)
    G.add_edge("E", "F", weight=10)
    G.add_edge("F", "G", weight=2)
    G.add_edge("G", "H", weight=1)
    G.add_edge("G", "I", weight=6)
    G.add_edge("H", "I", weight=7)

    # Find the shortest path from node A to node E
    path = nx.shortest_path(G, "A", "E", weight="weight")

    # Create a list of edges in the shortest path
    path_edges = list(zip(path, path[1:]))
    nx.set_node_attributes(G, nx.spring_layout(G, seed=37), "pos")
    nx.set_edge_attributes(
        G,
        {
            (u, v): {
                "color": (
                    "red"
                    if (u, v) in path_edges or tuple(reversed((u, v))) in path_edges
                    else "black"
                ),
                "label": {"label": d["weight"], "rotate": False},
            }
            for u, v, d in G.edges(data=True)
        },
    )
    nx.display(G, canvas=ax)
    plt.tight_layout()
    plt.axis("off")
    return fig


@pytest.mark.mpl_image_compare
def test_display_empty_graph():
    G = nx.empty_graph()
    fig, ax = plt.subplots()
    nx.display(G, canvas=ax)
    plt.tight_layout()
    plt.axis("off")
    return fig


@pytest.mark.mpl_image_compare
def test_house_with_colors():
    G = nx.house_graph()
    # explicitly set positions
    fig, ax = plt.subplots()
    pos = {0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (1, 1), 4: (0.5, 2.0)}

    # Plot nodes with different properties for the "wall" and "roof" nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=3000,
        nodelist=[0, 1, 2, 3],
        node_color="tab:blue",
    )
    nx.draw_networkx_nodes(
        G, pos, node_size=2000, nodelist=[4], node_color="tab:orange"
    )
    nx.draw_networkx_edges(G, pos, alpha=0.5, width=6)
    # Customize axes
    ax.margins(0.11)
    plt.tight_layout()
    plt.axis("off")
    return fig
