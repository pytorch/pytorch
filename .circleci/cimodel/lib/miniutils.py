def quote(s):
    return sandwich('"', s)


def sandwich(bread, jam):
    return bread + jam + bread


def override(word, substitutions):
    return substitutions.get(word, word)
