import importlib
import pickle

from setuptools import Distribution
import ordered_set


def test_reimport_extern():
    ordered_set2 = importlib.import_module(ordered_set.__name__)
    assert ordered_set is ordered_set2


def test_orderedset_pickle_roundtrip():
    o1 = ordered_set.OrderedSet([1, 2, 5])
    o2 = pickle.loads(pickle.dumps(o1))
    assert o1 == o2


def test_distribution_picklable():
    pickle.loads(pickle.dumps(Distribution()))
