#!/usr/bin/env python3
"""
model_dump: a one-stop shop for TorchScript model inspection.

The goal of this tool is to provide a simple way to extract lots of
useful information from a TorchScript model and make it easy for humans
to consume.  It (mostly) replaces zipinfo, common uses of show_pickle,
and various ad-hoc analysis notebooks.

The tool extracts information from the model and serializes it as JSON.
That JSON can then be rendered by an HTML+JS page, either by
loading the JSON over HTTP or producing a fully self-contained page
with all of the code and data burned-in.
"""

# Maintainer notes follow.
"""
The implementation strategy has tension between 3 goals:
- Small file size.
- Fully self-contained.
- Easy, modern JS environment.
Using Preact and HTM achieves 1 and 2 with a decent result for 3.
However, the models I tested with result in ~1MB JSON output,
so even using something heavier like full React might be tolerable
if the build process can be worked out.

One principle I have followed that I think is very beneficial
is to keep the JSON data as close as possible to the model
and do most of the rendering logic on the client.
This makes for easier development (just refresh, usually),
allows for more laziness and dynamism, and lets us add more
views of the same data without bloating the HTML file.

Currently, this code doesn't actually load the model or even
depend on any part of PyTorch.  I don't know if that's an important
feature to maintain, but it's probably worth preserving the ability
to run at least basic analysis on models that cannot be loaded.

I think the easiest way to develop this code is to cd into model_dump and
run "python -m http.server", then load http://localhost:8000/skeleton.html
in the browser.  In another terminal, run
"python -m torch.utils.model_dump --style=json FILE > \
    torch/utils/model_dump/model_info.json"
every time you update the Python code or model.
When you update JS, just refresh.

Possible improvements:
    - Fix various TODO comments in this file and the JS.
    - Make the HTML much less janky, especially the auxiliary data panel.
    - Make the auxiliary data panel start small, expand when
      data is available, and have a button to clear/contract.
    - Clean up the JS.  There's a lot of copypasta because
      I don't really know how to use Preact.
    - Make the HTML render and work nicely inside a Jupyter notebook.
    - Add the ability for JS to choose the URL to load the JSON based
      on the page URL (query or hash).  That way we could publish the
      inlined skeleton once and have it load various JSON blobs.
    - Add a button to expand all expandable sections so ctrl-F works well.
    - Add hyperlinking from data to code, and code to code.
    - Add hyperlinking from debug info to Diffusion.
    - Make small tensor contents available.
    - Do something nice for quantized models
      (they probably don't work at all right now).
"""

import sys
import os
import io
import pathlib
import re
import argparse
import zipfile
import json
import pickle
import pprint
import urllib.parse

from typing import (
    Dict,
)

import torch.utils.show_pickle


DEFAULT_EXTRA_FILE_SIZE_LIMIT = 16 * 1024

__all__ = ['get_storage_info', 'hierarchical_pickle', 'get_model_info', 'get_inline_skeleton',
           'burn_in_info', 'get_info_and_burn_skeleton']

def get_storage_info(storage):
    assert isinstance(storage, torch.utils.show_pickle.FakeObject)
    assert storage.module == "pers"
    assert storage.name == "obj"
    assert storage.state is None
    assert isinstance(storage.args, tuple)
    assert len(storage.args) == 1
    sa = storage.args[0]
    assert isinstance(sa, tuple)
    assert len(sa) == 5
    assert sa[0] == "storage"
    assert isinstance(sa[1], torch.utils.show_pickle.FakeClass)
    assert sa[1].module == "torch"
    assert sa[1].name.endswith("Storage")
    storage_info = [sa[1].name.replace("Storage", "")] + list(sa[2:])
    return storage_info


def hierarchical_pickle(data):
    if isinstance(data, (bool, int, float, str, type(None))):
        return data
    if isinstance(data, list):
        return [hierarchical_pickle(d) for d in data]
    if isinstance(data, tuple):
        return {
            "__tuple_values__": hierarchical_pickle(list(data)),
        }
    if isinstance(data, dict):
        return {
            "__is_dict__": True,
            "keys": hierarchical_pickle(list(data.keys())),
            "values": hierarchical_pickle(list(data.values())),
        }
    if isinstance(data, torch.utils.show_pickle.FakeObject):
        typename = f"{data.module}.{data.name}"
        if (
            typename.startswith(('__torch__.', 'torch.jit.LoweredWrapper.', 'torch.jit.LoweredModule.'))
        ):
            assert data.args == ()
            return {
                "__module_type__": typename,
                "state": hierarchical_pickle(data.state),
            }
        if typename == "torch._utils._rebuild_tensor_v2":
            assert data.state is None
            if len(data.args) == 6:
                storage, offset, size, stride, requires_grad, hooks = data.args
            else:
                storage, offset, size, stride, requires_grad, hooks, metadata = data.args
            storage_info = get_storage_info(storage)
            return {"__tensor_v2__": [storage_info, offset, size, stride, requires_grad]}
        if typename == "torch._utils._rebuild_qtensor":
            assert data.state is None
            storage, offset, size, stride, quantizer, requires_grad, hooks = data.args
            storage_info = get_storage_info(storage)
            assert isinstance(quantizer, tuple)
            assert isinstance(quantizer[0], torch.utils.show_pickle.FakeClass)
            assert quantizer[0].module == "torch"
            if quantizer[0].name == "per_tensor_affine":
                assert len(quantizer) == 3
                assert isinstance(quantizer[1], float)
                assert isinstance(quantizer[2], int)
                quantizer_extra = list(quantizer[1:3])
            else:
                quantizer_extra = []
            quantizer_json = [quantizer[0].name] + quantizer_extra
            return {"__qtensor__": [storage_info, offset, size, stride, quantizer_json, requires_grad]}
        if typename == "torch.jit._pickle.restore_type_tag":
            assert data.state is None
            obj, typ = data.args
            assert isinstance(typ, str)
            return hierarchical_pickle(obj)
        if re.fullmatch(r"torch\.jit\._pickle\.build_[a-z]+list", typename):
            assert data.state is None
            ls, = data.args
            assert isinstance(ls, list)
            return hierarchical_pickle(ls)
        if typename == "torch.device":
            assert data.state is None
            name, = data.args
            assert isinstance(name, str)
            # Just forget that it was a device and return the name.
            return name
        if typename == "builtin.UnicodeDecodeError":
            assert data.state is None
            msg, = data.args
            assert isinstance(msg, str)
            # Hack: Pretend this is a module so we don't need custom serialization.
            # Hack: Wrap the message in a tuple so it looks like a nice state object.
            # TODO: Undo at least that second hack.  We should support string states.
            return {
                "__module_type__": typename,
                "state": hierarchical_pickle((msg,)),
            }
        raise Exception(f"Can't prepare fake object of type for JS: {typename}")
    raise Exception(f"Can't prepare data of type for JS: {type(data)}")


def get_model_info(
        path_or_file,
        title=None,
        extra_file_size_limit=DEFAULT_EXTRA_FILE_SIZE_LIMIT):
    """Get JSON-friendly information about a model.

    The result is suitable for being saved as model_info.json,
    or passed to burn_in_info.
    """

    if isinstance(path_or_file, os.PathLike):
        default_title = os.fspath(path_or_file)
        file_size = path_or_file.stat().st_size  # type: ignore[attr-defined]
    elif isinstance(path_or_file, str):
        default_title = path_or_file
        file_size = pathlib.Path(path_or_file).stat().st_size
    else:
        default_title = "buffer"
        path_or_file.seek(0, io.SEEK_END)
        file_size = path_or_file.tell()
        path_or_file.seek(0)

    title = title or default_title

    with zipfile.ZipFile(path_or_file) as zf:
        path_prefix = None
        zip_files = []
        for zi in zf.infolist():
            prefix = re.sub("/.*", "", zi.filename)
            if path_prefix is None:
                path_prefix = prefix
            elif prefix != path_prefix:
                raise Exception(f"Mismatched prefixes: {path_prefix} != {prefix}")
            zip_files.append(dict(
                filename=zi.filename,
                compression=zi.compress_type,
                compressed_size=zi.compress_size,
                file_size=zi.file_size,
            ))

        assert path_prefix is not None
        version = zf.read(path_prefix + "/version").decode("utf-8").strip()

        def get_pickle(name):
            assert path_prefix is not None
            with zf.open(path_prefix + f"/{name}.pkl") as handle:
                raw = torch.utils.show_pickle.DumpUnpickler(handle, catch_invalid_utf8=True).load()
                return hierarchical_pickle(raw)

        model_data = get_pickle("data")
        constants = get_pickle("constants")

        # Intern strings that are likely to be re-used.
        # Pickle automatically detects shared structure,
        # so re-used strings are stored efficiently.
        # However, JSON has no way of representing this,
        # so we have to do it manually.
        interned_strings : Dict[str, int] = {}

        def ist(s):
            if s not in interned_strings:
                interned_strings[s] = len(interned_strings)
            return interned_strings[s]

        code_files = {}
        for zi in zf.infolist():
            if not zi.filename.endswith(".py"):
                continue
            with zf.open(zi) as handle:
                raw_code = handle.read()
            with zf.open(zi.filename + ".debug_pkl") as handle:
                raw_debug = handle.read()

            # Parse debug info and add begin/end markers if not present
            # to ensure that we cover the entire source code.
            debug_info_t = pickle.loads(raw_debug)
            text_table = None

            if (len(debug_info_t) == 3 and
                    isinstance(debug_info_t[0], str) and
                    debug_info_t[0] == 'FORMAT_WITH_STRING_TABLE'):
                _, text_table, content = debug_info_t

                def parse_new_format(line):
                    # (0, (('', '', 0), 0, 0))
                    num, ((text_indexes, fname_idx, offset), start, end), tag = line
                    text = ''.join(text_table[x] for x in text_indexes)  # type: ignore[index]
                    fname = text_table[fname_idx]  # type: ignore[index]
                    return num, ((text, fname, offset), start, end), tag

                debug_info_t = map(parse_new_format, content)

            debug_info = list(debug_info_t)
            if not debug_info:
                debug_info.append((0, (('', '', 0), 0, 0)))
            if debug_info[-1][0] != len(raw_code):
                debug_info.append((len(raw_code), (('', '', 0), 0, 0)))

            code_parts = []
            for di, di_next in zip(debug_info, debug_info[1:]):
                start, source_range, *_ = di
                end = di_next[0]
                assert end > start
                source, s_start, s_end = source_range
                s_text, s_file, s_line = source
                # TODO: Handle this case better.  TorchScript ranges are in bytes,
                # but JS doesn't really handle byte strings.
                # if bytes and chars are not equivalent for this string,
                # zero out the ranges so we don't highlight the wrong thing.
                if len(s_text) != len(s_text.encode("utf-8")):
                    s_start = 0
                    s_end = 0
                text = raw_code[start:end]
                code_parts.append([text.decode("utf-8"), ist(s_file), s_line, ist(s_text), s_start, s_end])
            code_files[zi.filename] = code_parts

        extra_files_json_pattern = re.compile(re.escape(path_prefix) + "/extra/.*\\.json")
        extra_files_jsons = {}
        for zi in zf.infolist():
            if not extra_files_json_pattern.fullmatch(zi.filename):
                continue
            if zi.file_size > extra_file_size_limit:
                continue
            with zf.open(zi) as handle:
                try:
                    json_content = json.load(handle)
                    extra_files_jsons[zi.filename] = json_content
                except json.JSONDecodeError:
                    extra_files_jsons[zi.filename] = "INVALID JSON"

        always_render_pickles = {
            "bytecode.pkl",
        }
        extra_pickles = {}
        for zi in zf.infolist():
            if not zi.filename.endswith(".pkl"):
                continue
            with zf.open(zi) as handle:
                # TODO: handle errors here and just ignore the file?
                # NOTE: For a lot of these files (like bytecode),
                # we could get away with just unpickling, but this should be safer.
                obj = torch.utils.show_pickle.DumpUnpickler(handle, catch_invalid_utf8=True).load()
            buf = io.StringIO()
            pprint.pprint(obj, buf)
            contents = buf.getvalue()
            # Checked the rendered length instead of the file size
            # because pickles with shared structure can explode in size during rendering.
            if os.path.basename(zi.filename) not in always_render_pickles and \
                    len(contents) > extra_file_size_limit:
                continue
            extra_pickles[zi.filename] = contents

    return {"model": dict(
        title=title,
        file_size=file_size,
        version=version,
        zip_files=zip_files,
        interned_strings=list(interned_strings),
        code_files=code_files,
        model_data=model_data,
        constants=constants,
        extra_files_jsons=extra_files_jsons,
        extra_pickles=extra_pickles,
    )}


def get_inline_skeleton():
    """Get a fully-inlined skeleton of the frontend.

    The returned HTML page has no external network dependencies for code.
    It can load model_info.json over HTTP, or be passed to burn_in_info.
    """

    import importlib.resources

    skeleton = importlib.resources.read_text(__package__, "skeleton.html")
    js_code = importlib.resources.read_text(__package__, "code.js")
    for js_module in ["preact", "htm"]:
        js_lib = importlib.resources.read_binary(__package__, f"{js_module}.mjs")
        js_url = "data:application/javascript," + urllib.parse.quote(js_lib)
        js_code = js_code.replace(f"https://unpkg.com/{js_module}?module", js_url)
    skeleton = skeleton.replace(' src="./code.js">', ">\n" + js_code)
    return skeleton


def burn_in_info(skeleton, info):
    """Burn model info into the HTML skeleton.

    The result will render the hard-coded model info and
    have no external network dependencies for code or data.
    """

    # Note that Python's json serializer does not escape slashes in strings.
    # Since we're inlining this JSON directly into a script tag, a string
    # containing "</script>" would end the script prematurely and
    # mess up our page.  Unconditionally escape fixes that.
    return skeleton.replace(
        "BURNED_IN_MODEL_INFO = null",
        "BURNED_IN_MODEL_INFO = " + json.dumps(info, sort_keys=True).replace("/", "\\/"))


def get_info_and_burn_skeleton(path_or_bytesio, **kwargs):
    model_info = get_model_info(path_or_bytesio, **kwargs)
    skeleton = get_inline_skeleton()
    page = burn_in_info(skeleton, model_info)
    return page


def main(argv, *, stdout=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--style", choices=["json", "html"])
    parser.add_argument("--title")
    parser.add_argument("model")
    args = parser.parse_args(argv[1:])

    info = get_model_info(args.model, title=args.title)

    output = stdout or sys.stdout

    if args.style == "json":
        output.write(json.dumps(info, sort_keys=True) + "\n")
    elif args.style == "html":
        skeleton = get_inline_skeleton()
        page = burn_in_info(skeleton, info)
        output.write(page)
    else:
        raise Exception("Invalid style")
