import json

import pytest

import networkx as nx
from networkx.readwrite.json_graph import node_link_data, node_link_graph


class TestNodeLink:
    def test_exception_dep(self):
        G = nx.MultiDiGraph()
        with pytest.raises(nx.NetworkXError):
            node_link_data(G, name="node", source="node", target="node", key="node")

    def test_graph(self):
        G = nx.path_graph(4)
        H = node_link_graph(node_link_data(G))
        assert nx.is_isomorphic(G, H)

    def test_graph_attributes(self):
        G = nx.path_graph(4)
        G.add_node(1, color="red")
        G.add_edge(1, 2, width=7)
        G.graph[1] = "one"
        G.graph["foo"] = "bar"

        H = node_link_graph(node_link_data(G))
        assert H.graph["foo"] == "bar"
        assert H.nodes[1]["color"] == "red"
        assert H[1][2]["width"] == 7

        d = json.dumps(node_link_data(G))
        H = node_link_graph(json.loads(d))
        assert H.graph["foo"] == "bar"
        assert H.graph["1"] == "one"
        assert H.nodes[1]["color"] == "red"
        assert H[1][2]["width"] == 7

    def test_digraph(self):
        G = nx.DiGraph()
        H = node_link_graph(node_link_data(G))
        assert H.is_directed()

    def test_multigraph(self):
        G = nx.MultiGraph()
        G.add_edge(1, 2, key="first")
        G.add_edge(1, 2, key="second", color="blue")
        H = node_link_graph(node_link_data(G))
        assert nx.is_isomorphic(G, H)
        assert H[1][2]["second"]["color"] == "blue"

    def test_graph_with_tuple_nodes(self):
        G = nx.Graph()
        G.add_edge((0, 0), (1, 0), color=[255, 255, 0])
        d = node_link_data(G)
        dumped_d = json.dumps(d)
        dd = json.loads(dumped_d)
        H = node_link_graph(dd)
        assert H.nodes[(0, 0)] == G.nodes[(0, 0)]
        assert H[(0, 0)][(1, 0)]["color"] == [255, 255, 0]

    def test_unicode_keys(self):
        q = "qualité"
        G = nx.Graph()
        G.add_node(1, **{q: q})
        s = node_link_data(G)
        output = json.dumps(s, ensure_ascii=False)
        data = json.loads(output)
        H = node_link_graph(data)
        assert H.nodes[1][q] == q

    def test_exception(self):
        G = nx.MultiDiGraph()
        attrs = {"name": "node", "source": "node", "target": "node", "key": "node"}
        with pytest.raises(nx.NetworkXError):
            node_link_data(G, **attrs)

    def test_string_ids(self):
        q = "qualité"
        G = nx.DiGraph()
        G.add_node("A")
        G.add_node(q)
        G.add_edge("A", q)
        data = node_link_data(G)
        assert data["edges"][0]["source"] == "A"
        assert data["edges"][0]["target"] == q
        H = node_link_graph(data)
        assert nx.is_isomorphic(G, H)

    def test_custom_attrs(self):
        G = nx.path_graph(4)
        G.add_node(1, color="red")
        G.add_edge(1, 2, width=7)
        G.graph[1] = "one"
        G.graph["foo"] = "bar"

        attrs = {
            "source": "c_source",
            "target": "c_target",
            "name": "c_id",
            "key": "c_key",
            "edges": "c_links",
        }

        H = node_link_graph(node_link_data(G, **attrs), multigraph=False, **attrs)
        assert nx.is_isomorphic(G, H)
        assert H.graph["foo"] == "bar"
        assert H.nodes[1]["color"] == "red"
        assert H[1][2]["width"] == 7
