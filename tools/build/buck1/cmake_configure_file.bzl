def cmake_configure_file(name: str, src: str, out: str, definitions: list[str]) -> None:
    if _type_name(definitions) == "buck_parser.select_support.SelectorList":
        definitions = _resolve_selector_list(definitions)

    native.genrule(
        name = name,
        out = out,
        srcs = [
            src,
        ],
        cmd = " ".join([
            "python3",
            "$(location //tools/build:cmake_configure_file.py)",
            "$SRCS",
            "$OUT",
        ] + definitions)
    )


def _type_name(o: object) -> str:
    """Gets the fully qualifed type name of an object.

    Buck 1 does not fully conform to the Starlark spec. In Starlark,
    the type() function is supposed to return a string identifying the
    type, but in Buck 1, it is just the Python `type()` builtin.

    This function is meant to get the type name as a string, since we
    need to identify types like
    buck_parser.select_support.SelectorValue that we can't get a
    reference to.

    Examples:
      * builtins.list
      * buck_parser.select_support.SelectorValue

    """
    t = type(o)
    return "{}.{}".format(t.__module__, t.__name__)


def _resolve_selector_list(selector) -> list[str]:
    assert _type_name(selector) ==  "buck_parser.select_support.SelectorList"
    ret = []
    for item in selector.items():
        if type(item) == type([]):
            ret.extend(item)
            continue
        assert _type_name(item) == "buck_parser.select_support.SelectorValue"

        # Hack: for right now, just resolve to the default
        # condition. But in the future, it might be nice to be able to
        # map to the Buck configuration instead, e.g. have some
        # mapping to the Buck 1 read_config() builtin.
        no_default = object()
        default = item.conditions().get("//conditions:default", no_default)
        if default == no_default:
            fail("Required \"//conditions:default\" in {}".format(item.conditions()))
        ret.extend(default)

    return ret
