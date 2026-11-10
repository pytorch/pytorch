"""The file ``torch.compiler.export_python`` writes: one graph's Inductor code, and one
compressed line holding the rest of the precompile artifact it was made from."""

import ast
import base64
import re
import textwrap
import zlib
from typing import cast

from torch._precompile import _parse_artifact_metadata, _source_literal, PrecompileError


# The file is a short header, the graph's Inductor code between these two lines, and
# one line holding the rest of the precompile artifact the capture wrote, compressed,
# with a placeholder where that artifact carried the graph code.
_CODE_BEGIN = "# " + "-" * 75 + " graph code"
_CODE_END = "# " + "-" * 71 + " end graph code"
_GRAPH_CODE = "__export_python_graph_code__"
_CAPTURED = re.compile(r"^_CAPTURED = '([A-Za-z0-9+/=]*)'$", re.MULTILINE)


def _comment(text: str) -> list[str]:
    return textwrap.wrap(text, 86, initial_indent="# ", subsequent_indent="# ")


def _exported_file(artifact: str) -> str:
    """The file export_python writes for the precompile ``artifact`` of one graph."""
    meta = _parse_artifact_metadata(artifact)
    name = f"{meta['ENTRY_MODULE']}.{meta['FN_NAME']}"
    sources = [
        node.value
        for node in ast.parse(artifact).body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Subscript)
        and getattr(node.targets[0].value, "id", None) == "GRAPH_SOURCES"
    ]
    if meta["SERVING_MODE"] != "standalone" or len(sources) > 1:
        raise PrecompileError(
            f"export_python: the capture of {name} is not one graph "
            f"(SERVING_MODE={meta['SERVING_MODE']!r}, {len(sources)} graphs)."
        )
    py = ".".join(map(str, cast(tuple[int, ...], meta["_DYNAMO_PYTHON_VERSION"])))
    lines = _comment(
        f"Exported by torch.compiler.export_python from {name}, one graph, captured "
        f"with torch {meta['TORCH_VERSION']} and Python {py}. The decorator serves "
        "this file; delete it to recapture."
    )
    if pinned := cast(tuple[str, ...], meta["WONT_GENERALIZE"]):
        lines += _comment(
            f"Served only with the captured values of: {', '.join(pinned)}."
        )
    if dropped := [n for _, n in cast(list[list[str]], meta["DROPPED_GUARDS"])]:
        lines += _comment(
            f"Not checked when served (they cannot be serialized): {', '.join(dropped)}."
        )
    if sources:
        lines += ["#"] + _comment(
            "The graph code below is Inductor's output and is what runs. Running this "
            "file benchmarks it on random inputs."
        )
        segment = ast.get_source_segment(artifact, sources[0])
        if segment is None or artifact.count(segment) != 1:
            raise AssertionError("the graph code literal is not unique in the artifact")
        artifact = artifact.replace(segment, _GRAPH_CODE)
        lines += ["", _CODE_BEGIN, ast.literal_eval(segment), _CODE_END]
    blob = base64.b64encode(zlib.compress(artifact.encode(), 9)).decode()
    lines += [""] + _comment(
        "The rest of what Dynamo captured (the guards, the frame's bytecode and "
        "AOTAutograd's wrapper around call()), compressed. Not meant to be edited."
    )
    return "\n".join([*lines, f"_CAPTURED = {blob!r}", ""])


def _captured_artifact(text: str, path: str) -> str:
    """The precompile artifact export_python's file ``text`` was written from, with the
    graph code the file holds now."""
    match = _CAPTURED.search(text)
    if match is None:
        raise PrecompileError(
            f"export_python: {path} has no _CAPTURED line, so it is not a file "
            "export_python wrote. Delete it to recapture."
        )
    artifact = zlib.decompress(base64.b64decode(match.group(1))).decode()
    if _GRAPH_CODE not in artifact:
        return artifact
    begin, end = text.find(_CODE_BEGIN + "\n"), text.rfind("\n" + _CODE_END)
    if begin < 0 or end < begin:
        raise PrecompileError(
            f"export_python: {path} lost the lines that delimit its graph code "
            f"({_CODE_BEGIN!r} and {_CODE_END!r}). Restore them, or delete the file "
            "to recapture."
        )
    code = text[begin + len(_CODE_BEGIN) + 1 : end]
    return artifact.replace(_GRAPH_CODE, _source_literal(code), 1)
