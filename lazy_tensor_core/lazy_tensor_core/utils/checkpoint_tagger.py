from __future__ import division
from __future__ import print_function

import collections
import json


class CheckpointTagger(object):

    def __init__(self, remover=None):
        self._tags = dict()
        self._refcount = collections.defaultdict(int)
        remover = (lambda x: None) if remover is None else remover
        assert callable(remover)
        self._remover = remover

    def tag(self, name, path):
        self._refcount[path] += 1
        old_path = self._tags.get(name)
        if old_path is not None:
            self._refcount[old_path] -= 1
            if self._refcount[old_path] == 0:
                self._refcount.pop(old_path)
                self._remover(old_path)
        self._tags[name] = path

    @property
    def tags(self):
        return self._tags

    def save_to_json(self):
        return json.dumps(self._tags)

    @classmethod
    def load_from_json(cls, str_json, remover=None):
        instance = cls(remover=remover)
        dat = json.loads(str_json)
        for name, path in dat.items():
            instance.tag(name, path)
        return instance
