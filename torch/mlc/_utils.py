class ReversibleDict(dict):
    def reversed(self):
        """
        Return a reversed dict, with common values in the original dict
        grouped into a list in the returned dict.

        Example:
        >>> d = ReversibleDict({'a': 3, 'c': 2, 'b': 2, 'e': 3, 'd': 1, 'f': 2})
        >>> d.reversed()
        {1: ['d'], 2: ['c', 'b', 'f'], 3: ['a', 'e']}
        """

        revdict = {}
        for k, v in self.items():
            revdict.setdefault(v, []).append(k)
        return revdict
