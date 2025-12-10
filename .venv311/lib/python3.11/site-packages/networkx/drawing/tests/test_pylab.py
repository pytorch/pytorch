"""Unit tests for matplotlib drawing functions."""

import itertools
import os
import warnings

import pytest

import networkx as nx

mpl = pytest.importorskip("matplotlib")
np = pytest.importorskip("numpy")
mpl.use("PS")
plt = pytest.importorskip("matplotlib.pyplot")
plt.rcParams["text.usetex"] = False


barbell = nx.barbell_graph(4, 6)

defaults = {
    "node_pos": None,
    "node_visible": True,
    "node_color": "#1f78b4",
    "node_size": 300,
    "node_label": {
        "size": 12,
        "color": "#000000",
        "family": "sans-serif",
        "weight": "normal",
        "alpha": 1.0,
        "background_color": None,
        "background_alpha": None,
        "h_align": "center",
        "v_align": "center",
        "bbox": None,
    },
    "node_shape": "o",
    "node_alpha": 1.0,
    "node_border_width": 1.0,
    "node_border_color": "face",
    "edge_visible": True,
    "edge_width": 1.0,
    "edge_color": "#000000",
    "edge_label": {
        "size": 12,
        "color": "#000000",
        "family": "sans-serif",
        "weight": "normal",
        "alpha": 1.0,
        "bbox": {"boxstyle": "round", "ec": (1.0, 1.0, 1.0), "fc": (1.0, 1.0, 1.0)},
        "h_align": "center",
        "v_align": "center",
        "pos": 0.5,
        "rotate": True,
    },
    "edge_style": "-",
    "edge_alpha": 1.0,
    # These are for undirected-graphs. Directed graphs shouls use "-|>" and 10, respectively
    "edge_arrowstyle": "-",
    "edge_arrowsize": 0,
    "edge_curvature": "arc3",
    "edge_source_margin": 0,
    "edge_target_margin": 0,
}


@pytest.mark.parametrize(
    ("param_name", "param_value", "expected"),
    (
        ("node_color", None, defaults["node_color"]),
        ("node_color", "#FF0000", "red"),
        ("node_color", "color", "lime"),
    ),
)
def test_display_arg_handling_node_color(param_name, param_value, expected):
    G = nx.path_graph(4)
    nx.set_node_attributes(G, "#00FF00", "color")
    canvas = plt.figure().add_subplot(111)
    nx.display(G, canvas=canvas, **{param_name: param_value})
    assert mpl.colors.same_color(canvas.get_children()[0].get_edgecolors()[0], expected)
    plt.close()


@pytest.mark.parametrize(
    ("param_value", "expected"),
    (
        (None, (1, 1, 1, 1)),  # default value
        (0.5, (0.5, 0.5, 0.5, 0.5)),
        ("n_alpha", (1.0, 1 / 2, 1 / 3, 0.25)),
    ),
)
def test_display_arg_handling_node_alpha(param_value, expected):
    G = nx.path_graph(4)
    nx.set_node_attributes(G, {n: 1 / (n + 1) for n in G.nodes()}, "n_alpha")
    canvas = plt.figure().add_subplot(111)
    nx.display(G, canvas=canvas, node_alpha=param_value)
    assert all(
        canvas.get_children()[0].get_fc()[:, 3] == expected
    )  # Extract just the alpha from the node colors
    plt.close()


def test_display_node_position():
    G = nx.path_graph(4)
    nx.set_node_attributes(G, {n: (n, n) for n in G.nodes()}, "pos")
    canvas = plt.figure().add_subplot(111)
    nx.display(G, canvas=canvas, node_pos="pos")
    assert np.all(
        canvas.get_children()[0].get_offsets().data == [[0, 0], [1, 1], [2, 2], [3, 3]]
    )
    plt.close()


def test_display_line_collection():
    G = nx.karate_club_graph()
    nx.set_edge_attributes(
        G, {(u, v): "-|>" if (u + v) % 2 else "-" for u, v in G.edges()}, "arrowstyle"
    )
    canvas = plt.figure().add_subplot(111)
    nx.display(G, canvas=canvas, edge_arrowsize=10)
    # There should only be one line collection in any given visualization
    lc = [
        l
        for l in canvas.get_children()
        if isinstance(l, mpl.collections.LineCollection)
    ][0]
    assert len(lc.get_paths()) == sum([1 for u, v in G.edges() if (u + v) % 2])
    plt.close()


@pytest.mark.parametrize(
    ("edge_color", "expected"),
    (
        (None, "black"),
        ("r", "red"),
        ((1.0, 1.0, 0.0), "yellow"),
        ((0, 1, 0, 1), "lime"),
        ("color", "blue"),
        ("#0000FF", "blue"),
    ),
)
@pytest.mark.parametrize("graph_type", (nx.Graph, nx.DiGraph))
def test_display_edge_single_color(edge_color, expected, graph_type):
    G = nx.path_graph(3, create_using=graph_type)
    nx.set_edge_attributes(G, "#0000FF", "color")
    canvas = plt.figure().add_subplot(111)
    nx.display(G, edge_color=edge_color, canvas=canvas)
    if G.is_directed():
        colors = [
            f.get_fc()
            for f in canvas.get_children()
            if isinstance(f, mpl.patches.FancyArrowPatch)
        ]
    else:
        colors = [
            l
            for l in canvas.collections
            if isinstance(l, mpl.collections.LineCollection)
        ][0].get_colors()
    assert all(mpl.colors.same_color(c, expected) for c in colors)
    plt.close()


@pytest.mark.parametrize("graph_type", (nx.Graph, nx.DiGraph))
def test_display_edge_multiple_colors(graph_type):
    G = nx.path_graph(3, create_using=graph_type)
    nx.set_edge_attributes(G, {(0, 1): "#FF0000", (1, 2): (0, 0, 1)}, "color")
    ax = plt.figure().add_subplot(111)
    nx.display(G, canvas=ax)
    expected = ["red", "blue"]
    if G.is_directed():
        colors = [
            f.get_fc()
            for f in ax.get_children()
            if isinstance(f, mpl.patches.FancyArrowPatch)
        ]
    else:
        colors = [
            l for l in ax.collections if isinstance(l, mpl.collections.LineCollection)
        ][0].get_colors()
    assert mpl.colors.same_color(colors, expected)
    plt.close()


@pytest.mark.parametrize("graph_type", (nx.Graph, nx.DiGraph))
def test_display_edge_position(graph_type):
    G = nx.path_graph(3, create_using=graph_type)
    nx.set_node_attributes(G, {n: (n, n) for n in G.nodes()}, "pos")
    ax = plt.figure().add_subplot(111)
    nx.display(G, canvas=ax)
    if G.is_directed():
        end_points = [
            (f.get_path().vertices[0, :], f.get_path().vertices[-2, :])
            for f in ax.get_children()
            if isinstance(f, mpl.patches.FancyArrowPatch)
        ]
    else:
        line_collection = [
            l for l in ax.collections if isinstance(l, mpl.collections.LineCollection)
        ][0]
        end_points = [
            (p.vertices[0, :], p.vertices[-1, :]) for p in line_collection.get_paths()
        ]
    expected = [((0, 0), (1, 1)), ((1, 1), (2, 2))]
    # Use the threshold to account for slight shifts in FancyArrowPatch margins to
    # avoid covering the arrow head with the node.
    threshold = 0.05
    for a, e in zip(end_points, expected):
        act_start, act_end = a
        exp_start, exp_end = e
        assert all(abs(act_start - exp_start) < (threshold, threshold)) and all(
            abs(act_end - exp_end) < (threshold, threshold)
        )
    plt.close()


def test_display_position_function():
    G = nx.karate_club_graph()

    def fixed_layout(G):
        return nx.spring_layout(G, seed=314159)

    pos = fixed_layout(G)
    ax = plt.figure().add_subplot(111)
    nx.display(G, node_pos=fixed_layout, canvas=ax)
    # rebuild the position dictionary from the canvas
    act_pos = {
        n: tuple(p) for n, p in zip(G.nodes(), ax.get_children()[0].get_offsets().data)
    }
    for n in G.nodes():
        assert all(pos[n] == act_pos[n])
    plt.close()


@pytest.mark.parametrize("graph_type", (nx.Graph, nx.DiGraph))
def test_display_edge_colormaps(graph_type):
    G = nx.path_graph(3, create_using=graph_type)
    nx.set_edge_attributes(G, {(0, 1): 0, (1, 2): 1}, "weight")
    cmap = mpl.colormaps["plasma"]
    nx.apply_matplotlib_colors(G, "weight", "color", cmap, nodes=False)
    canvas = plt.figure().add_subplot(111)
    nx.display(G, canvas=canvas)
    mapper = mpl.cm.ScalarMappable(cmap=cmap)
    mapper.set_clim(0, 1)
    expected = [mapper.to_rgba(0), mapper.to_rgba(1)]
    if G.is_directed():
        colors = [
            f.get_facecolor()
            for f in canvas.get_children()
            if isinstance(f, mpl.patches.FancyArrowPatch)
        ]
    else:
        colors = [
            l
            for l in canvas.collections
            if isinstance(l, mpl.collections.LineCollection)
        ][0].get_colors()
    assert mpl.colors.same_color(expected[0], G.edges[0, 1]["color"])
    assert mpl.colors.same_color(expected[1], G.edges[1, 2]["color"])
    assert mpl.colors.same_color(expected, colors)
    plt.close()


@pytest.mark.parametrize("graph_type", (nx.Graph, nx.DiGraph))
def test_display_node_colormaps(graph_type):
    G = nx.path_graph(3, create_using=graph_type)
    nx.set_node_attributes(G, {0: 0, 1: 0.5, 2: 1}, "weight")
    cmap = mpl.colormaps["plasma"]
    nx.apply_matplotlib_colors(G, "weight", "color", cmap)
    canvas = plt.figure().add_subplot(111)
    nx.display(G, canvas=canvas)
    mapper = mpl.cm.ScalarMappable(cmap=cmap)
    mapper.set_clim(0, 1)
    expected = [mapper.to_rgba(0), mapper.to_rgba(0.5), mapper.to_rgba(1)]
    colors = [
        s for s in canvas.collections if isinstance(s, mpl.collections.PathCollection)
    ][0].get_edgecolors()
    assert mpl.colors.same_color(expected[0], G.nodes[0]["color"])
    assert mpl.colors.same_color(expected[1], G.nodes[1]["color"])
    assert mpl.colors.same_color(expected[2], G.nodes[2]["color"])
    assert mpl.colors.same_color(expected, colors)
    plt.close()


@pytest.mark.parametrize(
    ("param_value", "expected"),
    (
        (None, [defaults["edge_width"], defaults["edge_width"]]),
        (5, [5, 5]),
        ("width", [5, 10]),
    ),
)
@pytest.mark.parametrize("graph_type", (nx.Graph, nx.DiGraph))
def test_display_edge_width(param_value, expected, graph_type):
    G = nx.path_graph(3, create_using=graph_type)
    nx.set_edge_attributes(G, {(0, 1): 5, (1, 2): 10}, "width")
    canvas = plt.figure().add_subplot(111)
    nx.display(G, edge_width=param_value, canvas=canvas)
    if G.is_directed():
        widths = [
            f.get_linewidth()
            for f in canvas.get_children()
            if isinstance(f, mpl.patches.FancyArrowPatch)
        ]
    else:
        widths = list(
            [
                l
                for l in canvas.collections
                if isinstance(l, mpl.collections.LineCollection)
            ][0].get_linewidths()
        )
    assert widths == expected


@pytest.mark.parametrize(
    ("param_value", "expected"),
    (
        (None, [defaults["edge_style"], defaults["edge_style"]]),
        (":", [":", ":"]),
        ("style", ["-", ":"]),
    ),
)
@pytest.mark.parametrize("graph_type", (nx.Graph, nx.DiGraph))
def test_display_edge_style(param_value, expected, graph_type):
    G = nx.path_graph(3, create_using=graph_type)
    nx.set_edge_attributes(G, {(0, 1): "-", (1, 2): ":"}, "style")
    canvas = plt.figure().add_subplot(111)
    nx.display(G, edge_style=param_value, canvas=canvas)
    if G.is_directed():
        styles = [
            f.get_linestyle()
            for f in canvas.get_children()
            if isinstance(f, mpl.patches.FancyArrowPatch)
        ]
    else:
        # Convert back from tuple description to character form
        linestyles = {(0, None): "-", (0, (1, 1.65)): ":"}
        styles = [
            linestyles[(s[0], tuple(s[1]) if s[1] is not None else None)]
            for s in [
                l
                for l in canvas.collections
                if isinstance(l, mpl.collections.LineCollection)
            ][0].get_linestyles()
        ]
    assert styles == expected
    plt.close()


def test_display_node_labels():
    G = nx.path_graph(4)
    canvas = plt.figure().add_subplot(111)
    nx.display(G, canvas=canvas, node_label={"size": 20})
    labels = [t for t in canvas.get_children() if isinstance(t, mpl.text.Text)]
    for n, l in zip(G.nodes(), labels):
        assert l.get_text() == str(n)
        assert l.get_size() == 20.0
    plt.close()


def test_display_edge_labels():
    G = nx.path_graph(4)
    canvas = plt.figure().add_subplot(111)
    # While we can pass in dicts for edge label defaults without errors,
    # this isn't helpful unless we want one label for all edges.
    nx.set_edge_attributes(G, {(u, v): {"label": u + v} for u, v in G.edges()})
    nx.display(G, canvas=canvas, edge_label={"color": "r"}, node_label=None)
    labels = [t for t in canvas.get_children() if isinstance(t, mpl.text.Text)]
    print(labels)
    for e, l in zip(G.edges(), labels):
        assert l.get_text() == str(e[0] + e[1])
        assert l.get_color() == "r"
    plt.close()


def test_display_multigraph_non_integer_keys():
    G = nx.MultiGraph()
    G.add_nodes_from(["A", "B", "C", "D"])
    G.add_edges_from(
        [
            ("A", "B", "0"),
            ("A", "B", "1"),
            ("B", "C", "-1"),
            ("B", "C", "1"),
            ("C", "D", "-1"),
            ("C", "D", "0"),
        ]
    )
    nx.set_edge_attributes(
        G, {e: f"arc3,rad={0.2 * int(e[2])}" for e in G.edges(keys=True)}, "curvature"
    )
    canvas = plt.figure().add_subplot(111)
    nx.display(G, canvas=canvas)
    rads = [
        f.get_connectionstyle().rad
        for f in canvas.get_children()
        if isinstance(f, mpl.patches.FancyArrowPatch)
    ]
    assert rads == [0.0, 0.2, -0.2, 0.2, -0.2, 0.0]
    plt.close()


def test_display_raises_for_bad_arg():
    G = nx.karate_club_graph()
    with pytest.raises(nx.NetworkXError):
        nx.display(G, bad_arg="bad_arg")
        plt.close()


def test_display_arrow_size():
    G = nx.path_graph(4, create_using=nx.DiGraph)
    nx.set_edge_attributes(
        G, {(u, v): (u + v + 2) ** 2 for u, v in G.edges()}, "arrowsize"
    )
    ax = plt.axes()
    nx.display(G, canvas=ax)
    assert [9, 25, 49] == [
        f.get_mutation_scale()
        for f in ax.get_children()
        if isinstance(f, mpl.patches.FancyArrowPatch)
    ]
    plt.close()


def test_display_mismatched_edge_position():
    """
    This test ensures that a error is raised for incomplete position data.
    """
    G = nx.path_graph(5)
    # Notice that there is no position for node 3
    nx.set_node_attributes(G, {0: (0, 0), 1: (1, 1), 2: (2, 2), 4: (4, 4)}, "pos")
    # But that's not a problem since we don't want to show node 4, right?
    nx.set_node_attributes(G, {n: n < 4 for n in G.nodes()}, "visible")
    # However, if we try to visualize every edge (including 3 -> 4)...
    # That's a problem since node 4 doesn't have a position
    with pytest.raises(nx.NetworkXError):
        nx.display(G)


# NOTE: parametrizing on marker to test both branches of internal
# to_marker_edge function
@pytest.mark.parametrize("node_shape", ("o", "s"))
def test_display_edge_margins(node_shape):
    """
    Test that there is a wider gap between the node and the start of an
    incident edge when min_source_margin is specified.

    This test checks that the use os min_{source/target}_margin edge
    attributes result is shorter (more padding) between the edges and
    source and target nodes.


    As a crude visual example, let 's' and 't' represent source and target
    nodes, respectively:

       Default:
       s-----------------------------t

       With margins:
       s   -----------------------   t

    """
    ax = plt.figure().add_subplot(111)
    G = nx.DiGraph([(0, 1)])
    nx.set_node_attributes(G, {0: (0, 0), 1: (1, 1)}, "pos")
    # Get the default patches from the regular visualization
    nx.display(G, canvas=ax, node_shape=node_shape)
    default_arrow = [
        f for f in ax.get_children() if isinstance(f, mpl.patches.FancyArrowPatch)
    ][0]
    default_extent = default_arrow.get_extents().corners()[::2, 0]
    # Now plot again with margins
    ax = plt.figure().add_subplot(111)
    nx.display(
        G,
        canvas=ax,
        edge_source_margin=100,
        edge_target_margin=100,
        node_shape=node_shape,
    )
    padded_arrow = [
        f for f in ax.get_children() if isinstance(f, mpl.patches.FancyArrowPatch)
    ][0]
    padded_extent = padded_arrow.get_extents().corners()[::2, 0]

    # With padding, the left-most extent of the edge should be further to the right
    assert padded_extent[0] > default_extent[0]
    # And the rightmost extent of the edge, further to the left
    assert padded_extent[1] < default_extent[1]
    plt.close()


@pytest.mark.parametrize("ticks", [False, True])
def test_display_hide_ticks(ticks):
    G = nx.path_graph(3)
    nx.set_node_attributes(G, {n: (n, n) for n in G.nodes()}, "pos")
    ax = plt.axes()
    nx.display(G, hide_ticks=ticks)
    for axis in [ax.xaxis, ax.yaxis]:
        assert bool(axis.get_ticklabels()) != ticks

    plt.close()


def test_display_self_loop():
    ax = plt.axes()
    G = nx.DiGraph()
    G.add_node(0)
    G.add_edge(0, 0)
    nx.set_node_attributes(G, {0: (0, 0)}, "pos")
    nx.display(G, canvas=ax)
    arrow = [
        f for f in ax.get_children() if isinstance(f, mpl.patches.FancyArrowPatch)
    ][0]
    bbox = arrow.get_extents()
    print(bbox.width)
    print(bbox.height)
    assert bbox.width > 0 and bbox.height > 0

    plt.delaxes(ax)
    plt.close()


def test_display_remove_pos_attr():
    """
    If the pos attribute isn't provided or is a function, display computes the layout
    and adds it to the graph. We need to ensure that this new attribute is removed from
    the returned graph.
    """
    G = nx.karate_club_graph()
    nx.display(G)
    assert nx.get_node_attributes(G, "display's position attribute name") == {}


@pytest.fixture
def subplots():
    fig, ax = plt.subplots()
    yield fig, ax
    plt.delaxes(ax)
    plt.close()


@pytest.mark.parametrize(
    "function",
    [
        nx.draw_circular,
        nx.draw_kamada_kawai,
        nx.draw_planar,
        nx.draw_random,
        nx.draw_spectral,
        nx.draw_spring,
        nx.draw_shell,
        nx.draw_forceatlas2,
    ],
)
def test_draw(function, subplots, tmp_path):
    if function == nx.draw_kamada_kawai:
        pytest.importorskip("scipy", reason="draw_kamada_kawai requires scipy")
    fig, _ = subplots
    options = {"node_color": "black", "node_size": 100, "width": 3}
    function(barbell, **options)
    fig.savefig(tmp_path / "test.ps")


def test_draw_shell_nlist(subplots, tmp_path):
    fig, _ = subplots
    nlist = [list(range(4)), list(range(4, 10)), list(range(10, 14))]
    nx.draw_shell(barbell, nlist=nlist)
    fig.savefig(tmp_path / "test.ps")


def test_draw_bipartite(subplots, tmp_path):
    fig, _ = subplots
    G = nx.complete_bipartite_graph(2, 5)
    nx.draw_bipartite(G)
    fig.savefig(tmp_path / "test.ps")


def test_edge_colormap():
    colors = range(barbell.number_of_edges())
    nx.draw_spring(
        barbell, edge_color=colors, width=4, edge_cmap=plt.cm.Blues, with_labels=True
    )
    # plt.show()


def test_draw_networkx_edge_labels(subplots, tmp_path):
    fig, _ = subplots
    edge = (0, 1)
    G = nx.DiGraph([edge])
    pos = {n: (n, n) for n in G}
    nx.draw(G, pos=pos)
    nx.draw_networkx_edge_labels(G, pos, edge_labels={edge: "edge"})
    fig.savefig(tmp_path / "test.ps")


def test_arrows():
    nx.draw_spring(barbell.to_directed())
    # plt.show()


@pytest.mark.parametrize(
    ("edge_color", "expected"),
    (
        (None, "black"),  # Default
        ("r", "red"),  # Non-default color string
        (["r"], "red"),  # Single non-default color in a list
        ((1.0, 1.0, 0.0), "yellow"),  # single color as rgb tuple
        ([(1.0, 1.0, 0.0)], "yellow"),  # single color as rgb tuple in list
        ((0, 1, 0, 1), "lime"),  # single color as rgba tuple
        ([(0, 1, 0, 1)], "lime"),  # single color as rgba tuple in list
        ("#0000ff", "blue"),  # single color hex code
        (["#0000ff"], "blue"),  # hex code in list
    ),
)
@pytest.mark.parametrize("edgelist", (None, [(0, 1)]))
def test_single_edge_color_undirected(edge_color, expected, edgelist):
    """Tests ways of specifying all edges have a single color for edges
    drawn with a LineCollection"""

    G = nx.path_graph(3)
    drawn_edges = nx.draw_networkx_edges(
        G, pos=nx.random_layout(G), edgelist=edgelist, edge_color=edge_color
    )
    assert mpl.colors.same_color(drawn_edges.get_color(), expected)


@pytest.mark.parametrize(
    ("edge_color", "expected"),
    (
        (None, "black"),  # Default
        ("r", "red"),  # Non-default color string
        (["r"], "red"),  # Single non-default color in a list
        ((1.0, 1.0, 0.0), "yellow"),  # single color as rgb tuple
        ([(1.0, 1.0, 0.0)], "yellow"),  # single color as rgb tuple in list
        ((0, 1, 0, 1), "lime"),  # single color as rgba tuple
        ([(0, 1, 0, 1)], "lime"),  # single color as rgba tuple in list
        ("#0000ff", "blue"),  # single color hex code
        (["#0000ff"], "blue"),  # hex code in list
    ),
)
@pytest.mark.parametrize("edgelist", (None, [(0, 1)]))
def test_single_edge_color_directed(edge_color, expected, edgelist):
    """Tests ways of specifying all edges have a single color for edges drawn
    with FancyArrowPatches"""

    G = nx.path_graph(3, create_using=nx.DiGraph)
    drawn_edges = nx.draw_networkx_edges(
        G, pos=nx.random_layout(G), edgelist=edgelist, edge_color=edge_color
    )
    for fap in drawn_edges:
        assert mpl.colors.same_color(fap.get_edgecolor(), expected)


def test_edge_color_tuple_interpretation():
    """If edge_color is a sequence with the same length as edgelist, then each
    value in edge_color is mapped onto each edge via colormap."""
    G = nx.path_graph(6, create_using=nx.DiGraph)
    pos = {n: (n, n) for n in range(len(G))}

    # num edges != 3 or 4 --> edge_color interpreted as rgb(a)
    for ec in ((0, 0, 1), (0, 0, 1, 1)):
        # More than 4 edges
        drawn_edges = nx.draw_networkx_edges(G, pos, edge_color=ec)
        for fap in drawn_edges:
            assert mpl.colors.same_color(fap.get_edgecolor(), ec)
        # Fewer than 3 edges
        drawn_edges = nx.draw_networkx_edges(
            G, pos, edgelist=[(0, 1), (1, 2)], edge_color=ec
        )
        for fap in drawn_edges:
            assert mpl.colors.same_color(fap.get_edgecolor(), ec)

    # num edges == 3, len(edge_color) == 4: interpreted as rgba
    drawn_edges = nx.draw_networkx_edges(
        G, pos, edgelist=[(0, 1), (1, 2), (2, 3)], edge_color=(0, 0, 1, 1)
    )
    for fap in drawn_edges:
        assert mpl.colors.same_color(fap.get_edgecolor(), "blue")

    # num edges == 4, len(edge_color) == 3: interpreted as rgb
    drawn_edges = nx.draw_networkx_edges(
        G, pos, edgelist=[(0, 1), (1, 2), (2, 3), (3, 4)], edge_color=(0, 0, 1)
    )
    for fap in drawn_edges:
        assert mpl.colors.same_color(fap.get_edgecolor(), "blue")

    # num edges == len(edge_color) == 3: interpreted with cmap, *not* as rgb
    drawn_edges = nx.draw_networkx_edges(
        G, pos, edgelist=[(0, 1), (1, 2), (2, 3)], edge_color=(0, 0, 1)
    )
    assert mpl.colors.same_color(
        drawn_edges[0].get_edgecolor(), drawn_edges[1].get_edgecolor()
    )
    for fap in drawn_edges:
        assert not mpl.colors.same_color(fap.get_edgecolor(), "blue")

    # num edges == len(edge_color) == 4: interpreted with cmap, *not* as rgba
    drawn_edges = nx.draw_networkx_edges(
        G, pos, edgelist=[(0, 1), (1, 2), (2, 3), (3, 4)], edge_color=(0, 0, 1, 1)
    )
    assert mpl.colors.same_color(
        drawn_edges[0].get_edgecolor(), drawn_edges[1].get_edgecolor()
    )
    assert mpl.colors.same_color(
        drawn_edges[2].get_edgecolor(), drawn_edges[3].get_edgecolor()
    )
    for fap in drawn_edges:
        assert not mpl.colors.same_color(fap.get_edgecolor(), "blue")


def test_fewer_edge_colors_than_num_edges_directed():
    """Test that the edge colors are cycled when there are fewer specified
    colors than edges."""
    G = barbell.to_directed()
    pos = nx.random_layout(barbell)
    edgecolors = ("r", "g", "b")
    drawn_edges = nx.draw_networkx_edges(G, pos, edge_color=edgecolors)
    for fap, expected in zip(drawn_edges, itertools.cycle(edgecolors)):
        assert mpl.colors.same_color(fap.get_edgecolor(), expected)


def test_more_edge_colors_than_num_edges_directed():
    """Test that extra edge colors are ignored when there are more specified
    colors than edges."""
    G = nx.path_graph(4, create_using=nx.DiGraph)  # 3 edges
    pos = nx.random_layout(barbell)
    edgecolors = ("r", "g", "b", "c")  # 4 edge colors
    drawn_edges = nx.draw_networkx_edges(G, pos, edge_color=edgecolors)
    for fap, expected in zip(drawn_edges, edgecolors[:-1]):
        assert mpl.colors.same_color(fap.get_edgecolor(), expected)


def test_edge_color_string_with_global_alpha_undirected():
    edge_collection = nx.draw_networkx_edges(
        barbell,
        pos=nx.random_layout(barbell),
        edgelist=[(0, 1), (1, 2)],
        edge_color="purple",
        alpha=0.2,
    )
    ec = edge_collection.get_color().squeeze()  # as rgba tuple
    assert len(edge_collection.get_paths()) == 2
    assert mpl.colors.same_color(ec[:-1], "purple")
    assert ec[-1] == 0.2


def test_edge_color_string_with_global_alpha_directed():
    drawn_edges = nx.draw_networkx_edges(
        barbell.to_directed(),
        pos=nx.random_layout(barbell),
        edgelist=[(0, 1), (1, 2)],
        edge_color="purple",
        alpha=0.2,
    )
    assert len(drawn_edges) == 2
    for fap in drawn_edges:
        ec = fap.get_edgecolor()  # As rgba tuple
        assert mpl.colors.same_color(ec[:-1], "purple")
        assert ec[-1] == 0.2


@pytest.mark.parametrize("graph_type", (nx.Graph, nx.DiGraph))
def test_edge_width_default_value(graph_type):
    """Test the default linewidth for edges drawn either via LineCollection or
    FancyArrowPatches."""
    G = nx.path_graph(2, create_using=graph_type)
    pos = {n: (n, n) for n in range(len(G))}
    drawn_edges = nx.draw_networkx_edges(G, pos)
    if isinstance(drawn_edges, list):  # directed case: list of FancyArrowPatch
        drawn_edges = drawn_edges[0]
    assert drawn_edges.get_linewidth() == 1


@pytest.mark.parametrize(
    ("edgewidth", "expected"),
    (
        (3, 3),  # single-value, non-default
        ([3], 3),  # Single value as a list
    ),
)
def test_edge_width_single_value_undirected(edgewidth, expected):
    G = nx.path_graph(4)
    pos = {n: (n, n) for n in range(len(G))}
    drawn_edges = nx.draw_networkx_edges(G, pos, width=edgewidth)
    assert len(drawn_edges.get_paths()) == 3
    assert drawn_edges.get_linewidth() == expected


@pytest.mark.parametrize(
    ("edgewidth", "expected"),
    (
        (3, 3),  # single-value, non-default
        ([3], 3),  # Single value as a list
    ),
)
def test_edge_width_single_value_directed(edgewidth, expected):
    G = nx.path_graph(4, create_using=nx.DiGraph)
    pos = {n: (n, n) for n in range(len(G))}
    drawn_edges = nx.draw_networkx_edges(G, pos, width=edgewidth)
    assert len(drawn_edges) == 3
    for fap in drawn_edges:
        assert fap.get_linewidth() == expected


@pytest.mark.parametrize(
    "edgelist",
    (
        [(0, 1), (1, 2), (2, 3)],  # one width specification per edge
        None,  #  fewer widths than edges - widths cycle
        [(0, 1), (1, 2)],  # More widths than edges - unused widths ignored
    ),
)
def test_edge_width_sequence(edgelist):
    G = barbell.to_directed()
    pos = nx.random_layout(G)
    widths = (0.5, 2.0, 12.0)
    drawn_edges = nx.draw_networkx_edges(G, pos, edgelist=edgelist, width=widths)
    for fap, expected_width in zip(drawn_edges, itertools.cycle(widths)):
        assert fap.get_linewidth() == expected_width


def test_edge_color_with_edge_vmin_vmax():
    """Test that edge_vmin and edge_vmax properly set the dynamic range of the
    color map when num edges == len(edge_colors)."""
    G = nx.path_graph(3, create_using=nx.DiGraph)
    pos = nx.random_layout(G)
    # Extract colors from the original (unscaled) colormap
    drawn_edges = nx.draw_networkx_edges(G, pos, edge_color=[0, 1.0])
    orig_colors = [e.get_edgecolor() for e in drawn_edges]
    # Colors from scaled colormap
    drawn_edges = nx.draw_networkx_edges(
        G, pos, edge_color=[0.2, 0.8], edge_vmin=0.2, edge_vmax=0.8
    )
    scaled_colors = [e.get_edgecolor() for e in drawn_edges]
    assert mpl.colors.same_color(orig_colors, scaled_colors)


def test_directed_edges_linestyle_default():
    """Test default linestyle for edges drawn with FancyArrowPatches."""
    G = nx.path_graph(4, create_using=nx.DiGraph)  # Graph with 3 edges
    pos = {n: (n, n) for n in range(len(G))}

    # edge with default style
    drawn_edges = nx.draw_networkx_edges(G, pos)
    assert len(drawn_edges) == 3
    for fap in drawn_edges:
        assert fap.get_linestyle() == "solid"


@pytest.mark.parametrize(
    "style",
    (
        "dashed",  # edge with string style
        "--",  # edge with simplified string style
        (1, (1, 1)),  # edge with (offset, onoffseq) style
    ),
)
def test_directed_edges_linestyle_single_value(style):
    """Tests support for specifying linestyles with a single value to be applied to
    all edges in ``draw_networkx_edges`` for FancyArrowPatch outputs
    (e.g. directed edges)."""

    G = nx.path_graph(4, create_using=nx.DiGraph)  # Graph with 3 edges
    pos = {n: (n, n) for n in range(len(G))}

    drawn_edges = nx.draw_networkx_edges(G, pos, style=style)
    assert len(drawn_edges) == 3
    for fap in drawn_edges:
        assert fap.get_linestyle() == style


@pytest.mark.parametrize(
    "style_seq",
    (
        ["dashed"],  # edge with string style in list
        ["--"],  # edge with simplified string style in list
        [(1, (1, 1))],  # edge with (offset, onoffseq) style in list
        ["--", "-", ":"],  # edges with styles for each edge
        ["--", "-"],  # edges with fewer styles than edges (styles cycle)
        ["--", "-", ":", "-."],  # edges with more styles than edges (extra unused)
    ),
)
def test_directed_edges_linestyle_sequence(style_seq):
    """Tests support for specifying linestyles with sequences in
    ``draw_networkx_edges`` for FancyArrowPatch outputs (e.g. directed edges)."""

    G = nx.path_graph(4, create_using=nx.DiGraph)  # Graph with 3 edges
    pos = {n: (n, n) for n in range(len(G))}

    drawn_edges = nx.draw_networkx_edges(G, pos, style=style_seq)
    assert len(drawn_edges) == 3
    for fap, style in zip(drawn_edges, itertools.cycle(style_seq)):
        assert fap.get_linestyle() == style


def test_return_types():
    from matplotlib.collections import LineCollection, PathCollection
    from matplotlib.patches import FancyArrowPatch

    G = nx.frucht_graph(create_using=nx.Graph)
    dG = nx.frucht_graph(create_using=nx.DiGraph)
    pos = nx.spring_layout(G, seed=42)
    dpos = nx.spring_layout(dG, seed=42)
    # nodes
    nodes = nx.draw_networkx_nodes(G, pos)
    assert isinstance(nodes, PathCollection)
    # edges
    edges = nx.draw_networkx_edges(dG, dpos, arrows=True)
    assert isinstance(edges, list)
    if len(edges) > 0:
        assert isinstance(edges[0], FancyArrowPatch)
    edges = nx.draw_networkx_edges(dG, dpos, arrows=False)
    assert isinstance(edges, LineCollection)
    edges = nx.draw_networkx_edges(G, dpos, arrows=None)
    assert isinstance(edges, LineCollection)
    edges = nx.draw_networkx_edges(dG, pos, arrows=None)
    assert isinstance(edges, list)
    if len(edges) > 0:
        assert isinstance(edges[0], FancyArrowPatch)


def test_labels_and_colors():
    G = nx.cubical_graph()
    pos = nx.spring_layout(G, seed=42)  # positions for all nodes
    # nodes
    nx.draw_networkx_nodes(
        G, pos, nodelist=[0, 1, 2, 3], node_color="r", node_size=500, alpha=0.75
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=[4, 5, 6, 7],
        node_color="b",
        node_size=500,
        alpha=[0.25, 0.5, 0.75, 1.0],
    )
    # edges
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=[(0, 1), (1, 2), (2, 3), (3, 0)],
        width=8,
        alpha=0.5,
        edge_color="r",
    )
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=[(4, 5), (5, 6), (6, 7), (7, 4)],
        width=8,
        alpha=0.5,
        edge_color="b",
    )
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=[(4, 5), (5, 6), (6, 7), (7, 4)],
        arrows=True,
        min_source_margin=0.5,
        min_target_margin=0.75,
        width=8,
        edge_color="b",
    )
    # some math labels
    labels = {}
    labels[0] = r"$a$"
    labels[1] = r"$b$"
    labels[2] = r"$c$"
    labels[3] = r"$d$"
    labels[4] = r"$\alpha$"
    labels[5] = r"$\beta$"
    labels[6] = r"$\gamma$"
    labels[7] = r"$\delta$"
    colors = {n: "k" if n % 2 == 0 else "r" for n in range(8)}
    nx.draw_networkx_labels(G, pos, labels, font_size=16)
    nx.draw_networkx_labels(G, pos, labels, font_size=16, font_color=colors)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=None, rotate=False)
    nx.draw_networkx_edge_labels(G, pos, edge_labels={(4, 5): "4-5"})
    # plt.show()


def test_axes(subplots):
    fig, ax = subplots
    nx.draw(barbell, ax=ax)
    nx.draw_networkx_edge_labels(barbell, nx.circular_layout(barbell), ax=ax)


def test_empty_graph():
    G = nx.Graph()
    nx.draw(G)


def test_draw_empty_nodes_return_values():
    # See Issue #3833
    import matplotlib.collections  # call as mpl.collections

    G = nx.Graph([(1, 2), (2, 3)])
    DG = nx.DiGraph([(1, 2), (2, 3)])
    pos = nx.circular_layout(G)
    assert isinstance(
        nx.draw_networkx_nodes(G, pos, nodelist=[]), mpl.collections.PathCollection
    )
    assert isinstance(
        nx.draw_networkx_nodes(DG, pos, nodelist=[]), mpl.collections.PathCollection
    )

    # drawing empty edges used to return an empty LineCollection or empty list.
    # Now it is always an empty list (because edges are now lists of FancyArrows)
    assert nx.draw_networkx_edges(G, pos, edgelist=[], arrows=True) == []
    assert nx.draw_networkx_edges(G, pos, edgelist=[], arrows=False) == []
    assert nx.draw_networkx_edges(DG, pos, edgelist=[], arrows=False) == []
    assert nx.draw_networkx_edges(DG, pos, edgelist=[], arrows=True) == []


def test_multigraph_edgelist_tuples():
    # See Issue #3295
    G = nx.path_graph(3, create_using=nx.MultiDiGraph)
    nx.draw_networkx(G, edgelist=[(0, 1, 0)])
    nx.draw_networkx(G, edgelist=[(0, 1, 0)], node_size=[10, 20, 0])


def test_alpha_iter():
    pos = nx.random_layout(barbell)
    fig = plt.figure()
    # with fewer alpha elements than nodes
    fig.add_subplot(131)  # Each test in a new axis object
    nx.draw_networkx_nodes(barbell, pos, alpha=[0.1, 0.2])
    # with equal alpha elements and nodes
    num_nodes = len(barbell.nodes)
    alpha = [x / num_nodes for x in range(num_nodes)]
    colors = range(num_nodes)
    fig.add_subplot(132)
    nx.draw_networkx_nodes(barbell, pos, node_color=colors, alpha=alpha)
    # with more alpha elements than nodes
    alpha.append(1)
    fig.add_subplot(133)
    nx.draw_networkx_nodes(barbell, pos, alpha=alpha)


def test_multiple_node_shapes(subplots):
    fig, ax = subplots
    G = nx.path_graph(4)
    nx.draw(G, node_shape=["o", "h", "s", "^"], ax=ax)
    scatters = [
        s for s in ax.get_children() if isinstance(s, mpl.collections.PathCollection)
    ]
    assert len(scatters) == 4


def test_individualized_font_attributes(subplots):
    G = nx.karate_club_graph()
    fig, ax = subplots
    nx.draw(
        G,
        ax=ax,
        font_color={n: "k" if n % 2 else "r" for n in G.nodes()},
        font_size={n: int(n / (34 / 15) + 5) for n in G.nodes()},
    )
    for n, t in zip(
        G.nodes(),
        [
            t
            for t in ax.get_children()
            if isinstance(t, mpl.text.Text) and len(t.get_text()) > 0
        ],
    ):
        expected = "black" if n % 2 else "red"

        assert mpl.colors.same_color(t.get_color(), expected)
        assert int(n / (34 / 15) + 5) == t.get_size()


def test_individualized_edge_attributes(subplots):
    G = nx.karate_club_graph()
    fig, ax = subplots
    arrowstyles = ["-|>" if (u + v) % 2 == 0 else "-[" for u, v in G.edges()]
    arrowsizes = [10 * (u % 2 + v % 2) + 10 for u, v in G.edges()]
    nx.draw(G, ax=ax, arrows=True, arrowstyle=arrowstyles, arrowsize=arrowsizes)
    arrows = [
        f for f in ax.get_children() if isinstance(f, mpl.patches.FancyArrowPatch)
    ]
    for e, a in zip(G.edges(), arrows):
        assert a.get_mutation_scale() == 10 * (e[0] % 2 + e[1] % 2) + 10
        expected = (
            mpl.patches.ArrowStyle.BracketB
            if sum(e) % 2
            else mpl.patches.ArrowStyle.CurveFilledB
        )
        assert isinstance(a.get_arrowstyle(), expected)


def test_error_invalid_kwds():
    with pytest.raises(ValueError, match="Received invalid argument"):
        nx.draw(barbell, foo="bar")


def test_draw_networkx_arrowsize_incorrect_size():
    G = nx.DiGraph([(0, 1), (0, 2), (0, 3), (1, 3)])
    arrowsize = [1, 2, 3]
    with pytest.raises(
        ValueError, match="arrowsize should have the same length as edgelist"
    ):
        nx.draw(G, arrowsize=arrowsize)


@pytest.mark.parametrize("arrowsize", (30, [10, 20, 30]))
def test_draw_edges_arrowsize(arrowsize):
    G = nx.DiGraph([(0, 1), (0, 2), (1, 2)])
    pos = {0: (0, 0), 1: (0, 1), 2: (1, 0)}
    edges = nx.draw_networkx_edges(G, pos=pos, arrowsize=arrowsize)

    arrowsize = itertools.repeat(arrowsize) if isinstance(arrowsize, int) else arrowsize

    for fap, expected in zip(edges, arrowsize):
        assert isinstance(fap, mpl.patches.FancyArrowPatch)
        assert fap.get_mutation_scale() == expected


@pytest.mark.parametrize("arrowstyle", ("-|>", ["-|>", "-[", "<|-|>"]))
def test_draw_edges_arrowstyle(arrowstyle):
    G = nx.DiGraph([(0, 1), (0, 2), (1, 2)])
    pos = {0: (0, 0), 1: (0, 1), 2: (1, 0)}
    edges = nx.draw_networkx_edges(G, pos=pos, arrowstyle=arrowstyle)

    arrowstyle = (
        itertools.repeat(arrowstyle) if isinstance(arrowstyle, str) else arrowstyle
    )

    arrow_objects = {
        "-|>": mpl.patches.ArrowStyle.CurveFilledB,
        "-[": mpl.patches.ArrowStyle.BracketB,
        "<|-|>": mpl.patches.ArrowStyle.CurveFilledAB,
    }

    for fap, expected in zip(edges, arrowstyle):
        assert isinstance(fap, mpl.patches.FancyArrowPatch)
        assert isinstance(fap.get_arrowstyle(), arrow_objects[expected])


def test_np_edgelist():
    # see issue #4129
    nx.draw_networkx(barbell, edgelist=np.array([(0, 2), (0, 3)]))


def test_draw_nodes_missing_node_from_position():
    G = nx.path_graph(3)
    pos = {0: (0, 0), 1: (1, 1)}  # No position for node 2
    with pytest.raises(nx.NetworkXError, match="has no position"):
        nx.draw_networkx_nodes(G, pos)


def test_draw_networkx_nodes_node_shape_list_with_scalar_color(subplots):
    """Ensure draw_networkx_nodes works when node_shape is a Python list.

    This covers the case where node_shape is a sequence (list) and node_color
    is a single scalar color, which should be supported.
    """
    fig, ax = subplots

    G = nx.empty_graph(5)
    pos = {i: (i, i) for i in G}

    shapes = ["o", "^", "o", "^", "o"]

    nodes = nx.draw_networkx_nodes(
        G,
        pos,
        node_color="red",  # scalar color (supported)
        node_shape=shapes,  # list of shapes – this used to be buggy
        ax=ax,
    )
    # Should get a PathCollection with an element in it (same as with numpy arrays)
    assert len(nodes.get_offsets()) > 0
    # NOTE: When node_shape is a sequence, draw_networkx_nodes internally calls
    # ax.scatter multiple times and returns only the last PathCollection.
    # Therefore, we do NOT assert a value for len(nodes.get_offsets()) here.


# NOTE: parametrizing on marker to test both branches of internal
# nx.draw_networkx_edges.to_marker_edge function
@pytest.mark.parametrize("node_shape", ("o", "s"))
def test_draw_edges_min_source_target_margins(node_shape, subplots):
    """Test that there is a wider gap between the node and the start of an
    incident edge when min_source_margin is specified.

    This test checks that the use of min_{source/target}_margin kwargs result
    in shorter (more padding) between the edges and source and target nodes.
    As a crude visual example, let 's' and 't' represent source and target
    nodes, respectively:

       Default:
       s-----------------------------t

       With margins:
       s   -----------------------   t

    """
    # Create a single axis object to get consistent pixel coords across
    # multiple draws
    fig, ax = subplots
    G = nx.DiGraph([(0, 1)])
    pos = {0: (0, 0), 1: (1, 0)}  # horizontal layout
    # Get leftmost and rightmost points of the FancyArrowPatch object
    # representing the edge between nodes 0 and 1 (in pixel coordinates)
    default_patch = nx.draw_networkx_edges(G, pos, ax=ax, node_shape=node_shape)[0]
    default_extent = default_patch.get_extents().corners()[::2, 0]
    # Now, do the same but with "padding" for the source and target via the
    # min_{source/target}_margin kwargs
    padded_patch = nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        node_shape=node_shape,
        min_source_margin=100,
        min_target_margin=100,
    )[0]
    padded_extent = padded_patch.get_extents().corners()[::2, 0]

    # With padding, the left-most extent of the edge should be further to the
    # right
    assert padded_extent[0] > default_extent[0]
    # And the rightmost extent of the edge, further to the left
    assert padded_extent[1] < default_extent[1]


# NOTE: parametrizing on marker to test both branches of internal
# nx.draw_networkx_edges.to_marker_edge function
@pytest.mark.parametrize("node_shape", ("o", "s"))
def test_draw_edges_min_source_target_margins_individual(node_shape, subplots):
    """Test that there is a wider gap between the node and the start of an
    incident edge when min_source_margin is specified.

    This test checks that the use of min_{source/target}_margin kwargs result
    in shorter (more padding) between the edges and source and target nodes.
    As a crude visual example, let 's' and 't' represent source and target
    nodes, respectively:

       Default:
       s-----------------------------t

       With margins:
       s   -----------------------   t

    """
    # Create a single axis object to get consistent pixel coords across
    # multiple draws
    fig, ax = subplots
    G = nx.DiGraph([(0, 1), (1, 2)])
    pos = {0: (0, 0), 1: (1, 0), 2: (2, 0)}  # horizontal layout
    # Get leftmost and rightmost points of the FancyArrowPatch object
    # representing the edge between nodes 0 and 1 (in pixel coordinates)
    default_patch = nx.draw_networkx_edges(G, pos, ax=ax, node_shape=node_shape)
    default_extent = [d.get_extents().corners()[::2, 0] for d in default_patch]
    # Now, do the same but with "padding" for the source and target via the
    # min_{source/target}_margin kwargs
    padded_patch = nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        node_shape=node_shape,
        min_source_margin=[98, 102],
        min_target_margin=[98, 102],
    )
    padded_extent = [p.get_extents().corners()[::2, 0] for p in padded_patch]
    for d, p in zip(default_extent, padded_extent):
        # With padding, the left-most extent of the edge should be further to the
        # right
        assert p[0] > d[0]
        # And the rightmost extent of the edge, further to the left
        assert p[1] < d[1]


def test_nonzero_selfloop_with_single_node(subplots):
    """Ensure that selfloop extent is non-zero when there is only one node."""
    # Create explicit axis object for test
    fig, ax = subplots
    # Graph with single node + self loop
    G = nx.DiGraph()
    G.add_node(0)
    G.add_edge(0, 0)
    # Draw
    patch = nx.draw_networkx_edges(G, {0: (0, 0)})[0]
    # The resulting patch must have non-zero extent
    bbox = patch.get_extents()
    assert bbox.width > 0 and bbox.height > 0


def test_nonzero_selfloop_with_single_edge_in_edgelist(subplots):
    """Ensure that selfloop extent is non-zero when only a single edge is
    specified in the edgelist.
    """
    # Create explicit axis object for test
    fig, ax = subplots
    # Graph with selfloop
    G = nx.path_graph(2, create_using=nx.DiGraph)
    G.add_edge(1, 1)
    pos = {n: (n, n) for n in G.nodes}
    # Draw only the selfloop edge via the `edgelist` kwarg
    patch = nx.draw_networkx_edges(G, pos, edgelist=[(1, 1)])[0]
    # The resulting patch must have non-zero extent
    bbox = patch.get_extents()
    assert bbox.width > 0 and bbox.height > 0


def test_apply_alpha():
    """Test apply_alpha when there is a mismatch between the number of
    supplied colors and elements.
    """
    nodelist = [0, 1, 2]
    colorlist = ["r", "g", "b"]
    alpha = 0.5
    rgba_colors = nx.drawing.nx_pylab.apply_alpha(colorlist, alpha, nodelist)
    assert all(rgba_colors[:, -1] == alpha)


def test_draw_edges_toggling_with_arrows_kwarg():
    """
    The `arrows` keyword argument is used as a 3-way switch to select which
    type of object to use for drawing edges:
      - ``arrows=None`` -> default (FancyArrowPatches for directed, else LineCollection)
      - ``arrows=True`` -> FancyArrowPatches
      - ``arrows=False`` -> LineCollection
    """
    import matplotlib.collections
    import matplotlib.patches

    UG = nx.path_graph(3)
    DG = nx.path_graph(3, create_using=nx.DiGraph)
    pos = {n: (n, n) for n in UG}

    # Use FancyArrowPatches when arrows=True, regardless of graph type
    for G in (UG, DG):
        edges = nx.draw_networkx_edges(G, pos, arrows=True)
        assert len(edges) == len(G.edges)
        assert isinstance(edges[0], mpl.patches.FancyArrowPatch)

    # Use LineCollection when arrows=False, regardless of graph type
    for G in (UG, DG):
        edges = nx.draw_networkx_edges(G, pos, arrows=False)
        assert isinstance(edges, mpl.collections.LineCollection)

    # Default behavior when arrows=None: FAPs for directed, LC's for undirected
    edges = nx.draw_networkx_edges(UG, pos)
    assert isinstance(edges, mpl.collections.LineCollection)
    edges = nx.draw_networkx_edges(DG, pos)
    assert len(edges) == len(G.edges)
    assert isinstance(edges[0], mpl.patches.FancyArrowPatch)


@pytest.mark.parametrize("drawing_func", (nx.draw, nx.draw_networkx))
def test_draw_networkx_arrows_default_undirected(drawing_func, subplots):
    import matplotlib.collections

    G = nx.path_graph(3)
    fig, ax = subplots
    drawing_func(G, ax=ax)
    assert any(isinstance(c, mpl.collections.LineCollection) for c in ax.collections)
    assert not ax.patches


@pytest.mark.parametrize("drawing_func", (nx.draw, nx.draw_networkx))
def test_draw_networkx_arrows_default_directed(drawing_func, subplots):
    import matplotlib.collections

    G = nx.path_graph(3, create_using=nx.DiGraph)
    fig, ax = subplots
    drawing_func(G, ax=ax)
    assert not any(
        isinstance(c, mpl.collections.LineCollection) for c in ax.collections
    )
    assert ax.patches


def test_edgelist_kwarg_not_ignored(subplots):
    # See gh-4994
    G = nx.path_graph(3)
    G.add_edge(0, 0)
    fig, ax = subplots
    nx.draw(G, edgelist=[(0, 1), (1, 2)], ax=ax)  # Exclude self-loop from edgelist
    assert not ax.patches


@pytest.mark.parametrize(
    ("G", "expected_n_edges"),
    ([nx.DiGraph(), 2], [nx.MultiGraph(), 4], [nx.MultiDiGraph(), 4]),
)
def test_draw_networkx_edges_multiedge_connectionstyle(G, expected_n_edges):
    """Draws edges correctly for 3 types of graphs and checks for valid length"""
    for i, (u, v) in enumerate([(0, 1), (0, 1), (0, 1), (0, 2)]):
        G.add_edge(u, v, weight=round(i / 3, 2))
    pos = {n: (n, n) for n in G}
    # Raises on insufficient connectionstyle length
    for conn_style in [
        "arc3,rad=0.1",
        ["arc3,rad=0.1", "arc3,rad=0.1"],
        ["arc3,rad=0.1", "arc3,rad=0.1", "arc3,rad=0.2"],
    ]:
        nx.draw_networkx_edges(G, pos, connectionstyle=conn_style)
        arrows = nx.draw_networkx_edges(G, pos, connectionstyle=conn_style)
        assert len(arrows) == expected_n_edges


@pytest.mark.parametrize(
    ("G", "expected_n_edges"),
    ([nx.DiGraph(), 2], [nx.MultiGraph(), 4], [nx.MultiDiGraph(), 4]),
)
def test_draw_networkx_edge_labels_multiedge_connectionstyle(G, expected_n_edges):
    """Draws labels correctly for 3 types of graphs and checks for valid length and class names"""
    for i, (u, v) in enumerate([(0, 1), (0, 1), (0, 1), (0, 2)]):
        G.add_edge(u, v, weight=round(i / 3, 2))
    pos = {n: (n, n) for n in G}
    # Raises on insufficient connectionstyle length
    arrows = nx.draw_networkx_edges(
        G, pos, connectionstyle=["arc3,rad=0.1", "arc3,rad=0.1", "arc3,rad=0.1"]
    )
    for conn_style in [
        "arc3,rad=0.1",
        ["arc3,rad=0.1", "arc3,rad=0.2"],
        ["arc3,rad=0.1", "arc3,rad=0.1", "arc3,rad=0.1"],
    ]:
        text_items = nx.draw_networkx_edge_labels(G, pos, connectionstyle=conn_style)
        assert len(text_items) == expected_n_edges
        for ti in text_items.values():
            assert ti.__class__.__name__ == "CurvedArrowText"


def test_draw_networkx_edge_label_multiedge():
    G = nx.MultiGraph()
    G.add_edge(0, 1, weight=10)
    G.add_edge(0, 1, weight=20)
    edge_labels = nx.get_edge_attributes(G, "weight")  # Includes edge keys
    pos = {n: (n, n) for n in G}
    text_items = nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        connectionstyle=["arc3,rad=0.1", "arc3,rad=0.2"],
    )
    assert len(text_items) == 2


def test_draw_networkx_edge_label_empty_dict():
    """Regression test for draw_networkx_edge_labels with empty dict. See
    gh-5372."""
    G = nx.path_graph(3)
    pos = {n: (n, n) for n in G.nodes}
    assert nx.draw_networkx_edge_labels(G, pos, edge_labels={}) == {}


def test_draw_networkx_edges_undirected_selfloop_colors(subplots):
    """When an edgelist is supplied along with a sequence of colors, check that
    the self-loops have the correct colors."""
    fig, ax = subplots
    # Edge list and corresponding colors
    edgelist = [(1, 3), (1, 2), (2, 3), (1, 1), (3, 3), (2, 2)]
    edge_colors = ["pink", "cyan", "black", "red", "blue", "green"]

    G = nx.Graph(edgelist)
    pos = {n: (n, n) for n in G.nodes}
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=edgelist, edge_color=edge_colors)

    # Verify that there are three fancy arrow patches (1 per self loop)
    assert len(ax.patches) == 3

    # These are points that should be contained in the self loops. For example,
    # sl_points[0] will be (1, 1.1), which is inside the "path" of the first
    # self-loop but outside the others
    sl_points = np.array(edgelist[-3:]) + np.array([0, 0.1])

    # Check that the mapping between self-loop locations and their colors is
    # correct
    for fap, clr, slp in zip(ax.patches, edge_colors[-3:], sl_points):
        assert fap.get_path().contains_point(slp)
        assert mpl.colors.same_color(fap.get_edgecolor(), clr)


@pytest.mark.parametrize(
    "fap_only_kwarg",  # Non-default values for kwargs that only apply to FAPs
    (
        {"arrowstyle": "-"},
        {"arrowsize": 20},
        {"connectionstyle": "arc3,rad=0.2"},
        {"min_source_margin": 10},
        {"min_target_margin": 10},
    ),
)
def test_user_warnings_for_unused_edge_drawing_kwargs(fap_only_kwarg, subplots):
    """Users should get a warning when they specify a non-default value for
    one of the kwargs that applies only to edges drawn with FancyArrowPatches,
    but FancyArrowPatches aren't being used under the hood."""
    G = nx.path_graph(3)
    pos = {n: (n, n) for n in G}
    fig, ax = subplots
    # By default, an undirected graph will use LineCollection to represent
    # the edges
    kwarg_name = list(fap_only_kwarg.keys())[0]
    with pytest.warns(
        UserWarning, match=f"\n\nThe {kwarg_name} keyword argument is not applicable"
    ):
        nx.draw_networkx_edges(G, pos, ax=ax, **fap_only_kwarg)
    # FancyArrowPatches are always used when `arrows=True` is specified.
    # Check that warnings are *not* raised in this case
    with warnings.catch_warnings():
        # Escalate warnings -> errors so tests fail if warnings are raised
        warnings.simplefilter("error")
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, **fap_only_kwarg)


@pytest.mark.parametrize("draw_fn", (nx.draw, nx.draw_circular))
def test_no_warning_on_default_draw_arrowstyle(draw_fn, subplots):
    # See gh-7284
    fig, ax = subplots
    G = nx.cycle_graph(5)
    with warnings.catch_warnings(record=True) as w:
        draw_fn(G, ax=ax)
    assert len(w) == 0


@pytest.mark.parametrize("hide_ticks", [False, True])
@pytest.mark.parametrize(
    "method",
    [
        nx.draw_networkx,
        nx.draw_networkx_edge_labels,
        nx.draw_networkx_edges,
        nx.draw_networkx_labels,
        nx.draw_networkx_nodes,
    ],
)
def test_hide_ticks(method, hide_ticks, subplots):
    G = nx.path_graph(3)
    pos = {n: (n, n) for n in G.nodes}
    _, ax = subplots
    method(G, pos=pos, ax=ax, hide_ticks=hide_ticks)
    for axis in [ax.xaxis, ax.yaxis]:
        assert bool(axis.get_ticklabels()) != hide_ticks


@pytest.mark.parametrize(
    "style", ["angle", "angle3", "arc", "arc3,rad=0.0", "bar,fraction=0.1"]
)
def test_edge_label_all_connectionstyles(subplots, style):
    """
    Check that FancyArrowPatches with all `connectionstyle`s are supported
    in edge label rendering. See gh-7735 and gh-8106.
    """
    fig, ax = subplots
    edge = (0, 1)
    G = nx.DiGraph([edge])
    pos = {n: (n, 0) for n in G}

    name = style.split(",")[0]
    labels = nx.draw_networkx_edge_labels(
        G, pos, edge_labels={edge: "edge"}, connectionstyle=style
    )

    hmid = (pos[0][0] + pos[1][0]) / 2
    vmid = (pos[0][1] + pos[1][1]) / 2
    if name in {"arc", "arc3"}:  # The label should be at roughly the midpoint.
        assert labels[edge].x, labels[edge].y == pytest.approx((hmid, vmid))
    elif name == "bar":  # The label should be below the vertical midpoint.
        assert labels[edge].y < vmid


@pytest.mark.parametrize("label_pos", [-0.1, 1.1])
def test_edge_label_label_pos(subplots, label_pos):
    """
    Check that label positions can be extrapolated outside [0, 1].
    """
    fig, ax = subplots
    edge = (0, 1)
    G = nx.DiGraph([edge])
    pos = {n: (n, n) for n in G}
    lbl = nx.draw_networkx_edge_labels(
        G, pos, edge_labels={edge: "edge"}, label_pos=label_pos, connectionstyle="angle"
    )

    assert lbl[edge].x, lbl[edge].y == pytest.approx((label_pos, label_pos))
