import networkx as nx


def test_chordal_graph():
    G = nx.complete_graph(5)
    assert nx.is_perfect_graph(G)


def test_odd_cycle():
    G = nx.cycle_graph(5)  # Induced odd cycle
    assert not nx.is_perfect_graph(G)


def test_even_cycle():
    G = nx.cycle_graph(6)  # Even cycle is perfect
    assert nx.is_perfect_graph(G)


def test_complement_of_odd_cycle():
    G = nx.cycle_graph(7)
    GC = nx.complement(G)
    assert not nx.is_perfect_graph(GC)


def test_disconnected_union_of_cliques():
    G = nx.disjoint_union(nx.complete_graph(3), nx.complete_graph(4))
    assert nx.is_perfect_graph(G)
