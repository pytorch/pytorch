"""
Tests for closeness centrality.
"""

import pytest

import networkx as nx


@pytest.fixture()
def undirected_G():
    G = nx.fast_gnp_random_graph(n=100, p=0.6, seed=123)
    cc = nx.closeness_centrality(G)
    return G, cc


class TestClosenessCentrality:
    def test_wf_improved(self):
        G = nx.union(nx.path_graph(4), nx.path_graph([4, 5, 6]))
        c = nx.closeness_centrality(G)
        cwf = nx.closeness_centrality(G, wf_improved=False)
        res = {0: 0.25, 1: 0.375, 2: 0.375, 3: 0.25, 4: 0.222, 5: 0.333, 6: 0.222}
        wf_res = {0: 0.5, 1: 0.75, 2: 0.75, 3: 0.5, 4: 0.667, 5: 1.0, 6: 0.667}
        for n in G:
            assert c[n] == pytest.approx(res[n], abs=1e-3)
            assert cwf[n] == pytest.approx(wf_res[n], abs=1e-3)

    def test_digraph(self):
        G = nx.path_graph(3, create_using=nx.DiGraph)
        c = nx.closeness_centrality(G)
        cr = nx.closeness_centrality(G.reverse())
        d = {0: 0.0, 1: 0.500, 2: 0.667}
        dr = {0: 0.667, 1: 0.500, 2: 0.0}
        for n in G:
            assert c[n] == pytest.approx(d[n], abs=1e-3)
            assert cr[n] == pytest.approx(dr[n], abs=1e-3)

    def test_k5_closeness(self):
        G = nx.complete_graph(5)
        c = nx.closeness_centrality(G)
        d = {0: 1.000, 1: 1.000, 2: 1.000, 3: 1.000, 4: 1.000}
        for n in G:
            assert c[n] == pytest.approx(d[n], abs=1e-3)

    def test_p3_closeness(self):
        G = nx.path_graph(3)
        c = nx.closeness_centrality(G)
        d = {0: 0.667, 1: 1.000, 2: 0.667}
        for n in G:
            assert c[n] == pytest.approx(d[n], abs=1e-3)

    def test_krackhardt_closeness(self):
        G = nx.krackhardt_kite_graph()
        c = nx.closeness_centrality(G)
        d = {
            0: 0.529,
            1: 0.529,
            2: 0.500,
            3: 0.600,
            4: 0.500,
            5: 0.643,
            6: 0.643,
            7: 0.600,
            8: 0.429,
            9: 0.310,
        }
        for n in G:
            assert c[n] == pytest.approx(d[n], abs=1e-3)

    def test_florentine_families_closeness(self):
        G = nx.florentine_families_graph()
        c = nx.closeness_centrality(G)
        d = {
            "Acciaiuoli": 0.368,
            "Albizzi": 0.483,
            "Barbadori": 0.4375,
            "Bischeri": 0.400,
            "Castellani": 0.389,
            "Ginori": 0.333,
            "Guadagni": 0.467,
            "Lamberteschi": 0.326,
            "Medici": 0.560,
            "Pazzi": 0.286,
            "Peruzzi": 0.368,
            "Ridolfi": 0.500,
            "Salviati": 0.389,
            "Strozzi": 0.4375,
            "Tornabuoni": 0.483,
        }
        for n in G:
            assert c[n] == pytest.approx(d[n], abs=1e-3)

    def test_les_miserables_closeness(self):
        G = nx.les_miserables_graph()
        c = nx.closeness_centrality(G)
        d = {
            "Napoleon": 0.302,
            "Myriel": 0.429,
            "MlleBaptistine": 0.413,
            "MmeMagloire": 0.413,
            "CountessDeLo": 0.302,
            "Geborand": 0.302,
            "Champtercier": 0.302,
            "Cravatte": 0.302,
            "Count": 0.302,
            "OldMan": 0.302,
            "Valjean": 0.644,
            "Labarre": 0.394,
            "Marguerite": 0.413,
            "MmeDeR": 0.394,
            "Isabeau": 0.394,
            "Gervais": 0.394,
            "Listolier": 0.341,
            "Tholomyes": 0.392,
            "Fameuil": 0.341,
            "Blacheville": 0.341,
            "Favourite": 0.341,
            "Dahlia": 0.341,
            "Zephine": 0.341,
            "Fantine": 0.461,
            "MmeThenardier": 0.461,
            "Thenardier": 0.517,
            "Cosette": 0.478,
            "Javert": 0.517,
            "Fauchelevent": 0.402,
            "Bamatabois": 0.427,
            "Perpetue": 0.318,
            "Simplice": 0.418,
            "Scaufflaire": 0.394,
            "Woman1": 0.396,
            "Judge": 0.404,
            "Champmathieu": 0.404,
            "Brevet": 0.404,
            "Chenildieu": 0.404,
            "Cochepaille": 0.404,
            "Pontmercy": 0.373,
            "Boulatruelle": 0.342,
            "Eponine": 0.396,
            "Anzelma": 0.352,
            "Woman2": 0.402,
            "MotherInnocent": 0.398,
            "Gribier": 0.288,
            "MmeBurgon": 0.344,
            "Jondrette": 0.257,
            "Gavroche": 0.514,
            "Gillenormand": 0.442,
            "Magnon": 0.335,
            "MlleGillenormand": 0.442,
            "MmePontmercy": 0.315,
            "MlleVaubois": 0.308,
            "LtGillenormand": 0.365,
            "Marius": 0.531,
            "BaronessT": 0.352,
            "Mabeuf": 0.396,
            "Enjolras": 0.481,
            "Combeferre": 0.392,
            "Prouvaire": 0.357,
            "Feuilly": 0.392,
            "Courfeyrac": 0.400,
            "Bahorel": 0.394,
            "Bossuet": 0.475,
            "Joly": 0.394,
            "Grantaire": 0.358,
            "MotherPlutarch": 0.285,
            "Gueulemer": 0.463,
            "Babet": 0.463,
            "Claquesous": 0.452,
            "Montparnasse": 0.458,
            "Toussaint": 0.402,
            "Child1": 0.342,
            "Child2": 0.342,
            "Brujon": 0.380,
            "MmeHucheloup": 0.353,
        }
        for n in G:
            assert c[n] == pytest.approx(d[n], abs=1e-3)

    def test_weighted_closeness(self):
        edges = [
            ("s", "u", 10),
            ("s", "x", 5),
            ("u", "v", 1),
            ("u", "x", 2),
            ("v", "y", 1),
            ("x", "u", 3),
            ("x", "v", 5),
            ("x", "y", 2),
            ("y", "s", 7),
            ("y", "v", 6),
        ]
        XG = nx.Graph()
        XG.add_weighted_edges_from(edges)
        c = nx.closeness_centrality(XG, distance="weight")
        d = {"y": 0.200, "x": 0.286, "s": 0.138, "u": 0.235, "v": 0.200}
        for n in sorted(XG):
            assert c[n] == pytest.approx(d[n], abs=1e-3)


class TestIncrementalClosenessCentrality:
    @staticmethod
    def pick_add_edge(G):
        u = nx.utils.arbitrary_element(G)
        possible_nodes = set(G) - (set(G.neighbors(u)) | {u})
        v = nx.utils.arbitrary_element(possible_nodes)
        return (u, v)

    @staticmethod
    def pick_remove_edge(G):
        u = nx.utils.arbitrary_element(G)
        possible_nodes = list(G.neighbors(u))
        v = nx.utils.arbitrary_element(possible_nodes)
        return (u, v)

    def test_directed_raises(self):
        dir_G = nx.gn_graph(n=5)
        prev_cc = None
        edge = self.pick_add_edge(dir_G)
        with pytest.raises(nx.NetworkXNotImplemented):
            nx.incremental_closeness_centrality(dir_G, edge, prev_cc, insertion=True)

    def test_wrong_size_prev_cc_raises(self, undirected_G):
        G, prev_cc = undirected_G
        edge = self.pick_add_edge(G)
        prev_cc.pop(0)
        with pytest.raises(nx.NetworkXError):
            nx.incremental_closeness_centrality(G, edge, prev_cc, insertion=True)

    def test_wrong_nodes_prev_cc_raises(self, undirected_G):
        G, prev_cc = undirected_G

        edge = self.pick_add_edge(G)
        num_nodes = len(prev_cc)
        prev_cc.pop(0)
        prev_cc[num_nodes] = 0.5
        with pytest.raises(nx.NetworkXError):
            nx.incremental_closeness_centrality(G, edge, prev_cc, insertion=True)

    def test_zero_centrality(self):
        G = nx.path_graph(3)
        prev_cc = nx.closeness_centrality(G)
        edge = self.pick_remove_edge(G)
        test_cc = nx.incremental_closeness_centrality(G, edge, prev_cc, insertion=False)
        G.remove_edges_from([edge])
        real_cc = nx.closeness_centrality(G)
        shared_items = set(test_cc.items()) & set(real_cc.items())
        assert len(shared_items) == len(real_cc)
        assert 0 in test_cc.values()

    def test_incremental(self, undirected_G):
        # Check that incremental and regular give same output
        G, _ = undirected_G
        prev_cc = None
        for i in range(5):
            if i % 2 == 0:
                # Remove an edge
                insert = False
                edge = self.pick_remove_edge(G)
            else:
                # Add an edge
                insert = True
                edge = self.pick_add_edge(G)

            test_cc = nx.incremental_closeness_centrality(G, edge, prev_cc, insert)

            if insert:
                G.add_edges_from([edge])
            else:
                G.remove_edges_from([edge])

            real_cc = nx.closeness_centrality(G)

            assert set(test_cc.items()) == set(real_cc.items())

            prev_cc = test_cc
