def quote(s):
    return '"' + s + '"'


def override(word, substitutions):
    return substitutions.get(word, word)
