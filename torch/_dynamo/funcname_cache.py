import tokenize
from typing import Dict, List, Optional


cache: Dict[str, Dict[int, str]] = {}


def clearcache() -> None:
    cache.clear()


def _add_file(filename: str) -> None:
    try:
        with tokenize.open(filename) as f:
            tokens = list(tokenize.generate_tokens(f.readline))
    except OSError:
        cache[filename] = {}
        return

    # NOTE: undefined behavior if file is not valid Python source,
    # since tokenize will have undefined behavior.
    result: Dict[int, str] = {}
    # current full funcname, e.g. xxx.yyy.zzz
    cur_name = ""
    cur_indent = 0
    significant_indents: List[int] = []

    for i, token in enumerate(tokens):
        if token.type == tokenize.INDENT:
            cur_indent += 1
        elif token.type == tokenize.DEDENT:
            cur_indent -= 1
            # possible end of function or class
            if significant_indents and cur_indent == significant_indents[-1]:
                significant_indents.pop()
                # pop the last name
                cur_name = cur_name.rpartition(".")[0]
        elif (
            token.type == tokenize.NAME
            and i + 1 < len(tokens)
            and tokens[i + 1].type == tokenize.NAME
            and (token.string == "class" or token.string == "def")
        ):
            # name of class/function always follows class/def token
            significant_indents.append(cur_indent)
            if cur_name:
                cur_name += "."
            cur_name += tokens[i + 1].string
        result[token.start[0]] = cur_name

    cache[filename] = result


def get_funcname(filename: str, lineno: int) -> Optional[str]:
    if filename not in cache:
        _add_file(filename)
    return cache[filename].get(lineno, None)
