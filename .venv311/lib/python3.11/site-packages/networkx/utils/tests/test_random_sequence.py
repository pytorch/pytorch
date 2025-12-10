import pytest

import networkx as nx


def test_degree_sequences():
    seq = nx.utils.powerlaw_sequence(10, seed=1)
    seq = nx.utils.powerlaw_sequence(10)
    assert len(seq) == 10


@pytest.mark.parametrize(
    ("deg_seq", "valid", "reason"),
    [
        ([], False, "must have one more node"),
        ([0], True, ""),
        ([2], False, "must have one more node"),
        ([2, 0], False, "must have strictly positive"),
        ([3, 1, 1, 1], True, ""),
    ],
)
def test_valid_degree_sequence(deg_seq, valid, reason):
    v, r = nx.utils.is_valid_tree_degree_sequence(deg_seq)
    assert v == valid
    assert reason in r


def test_zipf_rv():
    r = nx.utils.zipf_rv(2.3, xmin=2, seed=1)
    r = nx.utils.zipf_rv(2.3, 2, 1)
    r = nx.utils.zipf_rv(2.3)
    assert type(r), int
    pytest.raises(ValueError, nx.utils.zipf_rv, 0.5)
    pytest.raises(ValueError, nx.utils.zipf_rv, 2, xmin=0)


def test_random_weighted_sample():
    mapping = {"a": 10, "b": 20}
    s = nx.utils.random_weighted_sample(mapping, 2, seed=1)
    s = nx.utils.random_weighted_sample(mapping, 2)
    assert sorted(s) == sorted(mapping.keys())
    pytest.raises(ValueError, nx.utils.random_weighted_sample, mapping, 3)


def test_random_weighted_choice():
    mapping = {"a": 10, "b": 0}
    c = nx.utils.weighted_choice(mapping, seed=1)
    c = nx.utils.weighted_choice(mapping)
    assert c == "a"


def test_random_sequence_low_precision():
    assert nx.utils.cumulative_distribution([0.1] * 100)[-1] == 1.0
