import collections
import contextlib
import cProfile
import functools
import itertools
import logging
import os.path
import pstats
import shutil
import subprocess
from typing import Any, List
from unittest.mock import patch

from functorch.compile import draw_graph, get_aot_graph_name, get_graph_being_compiled

import torch
from torch import fx as fx

from torch._dynamo.repro.after_aot import save_graph_repro, wrap_compiler_debug
from torch._dynamo.utils import get_debug_dir
from torch.fx.graph_module import GraphModule
from torch.fx.passes.shape_prop import TensorMetadata
from torch.fx.passes.tools_common import legalize_graph

from . import config, ir  # noqa: F811, this is needed
from .scheduler import (
    BaseSchedulerNode,
    FusedSchedulerNode,
    NopKernelSchedulerNode,
    OutputNode,
    SchedulerNode,
)
from .virtualized import V

log = logging.getLogger(__name__)


@functools.lru_cache(None)
def has_dot():
    try:
        subprocess.check_output(["which", "dot"], stderr=subprocess.PIPE)
        return True
    except subprocess.SubprocessError:
        return False


def draw_buffers(nodes, print_graph=False, fname=None):
    """
    Draw a graph in fname.svg.
    nodes is a list of SchedulerNode objects.
    """
    if not has_dot():
        log.warning("draw_buffers() requires `graphviz` package")
        return

    if fname is None:
        fname = get_graph_being_compiled()

    graph = create_fx_from_snodes(nodes)

    for node in graph.nodes:
        if "fusion_meta" not in node.meta:
            continue
        group = node.meta["fusion_meta"].group
        if isinstance(group, tuple):
            group = group[1]

        # gather meta data
        dtype = None
        if isinstance(node, ir.ComputedBuffer):
            dtype = node.data.dtype

        metadata = TensorMetadata(group, dtype, None, None, None, None, None)
        node.meta["tensor_meta"] = metadata

    if print_graph:
        print(graph)

    gm = GraphModule({}, graph)
    legalize_graph(gm)
    gm.graph.lint()
    draw_graph(gm, fname, clear_meta=False)


def create_fx_from_snodes(snodes: List[BaseSchedulerNode]) -> fx.Graph:
    """
    Creates a FX Graph from a list of SchedulerNode objects.
    """

    def get_fake_func(name):
        def func1(*args):
            return 0

        func1.__name__ = name
        return func1

    FusionMeta = collections.namedtuple("FusionMeta", ["group", "snode", "type"])

    func_dict = {s: get_fake_func(s) for s in ["extern", "nop", "compute", "fused"]}
    buf_to_fx_node = {}
    graph = torch.fx.Graph()
    first_node = None

    outputs = []
    group: Any = None
    # create call_function node for each Buffer and Kernel
    for snode in snodes:
        if snode.is_extern():
            node_type = "extern"
            group = node_type
        elif snode.is_template():
            node_type = "template"
            group = node_type
        elif isinstance(snode, NopKernelSchedulerNode):
            node_type = "nop"
            group = node_type
        elif isinstance(snode, SchedulerNode):
            node_type = "compute"
            group = snode.group
        elif isinstance(snode, FusedSchedulerNode):
            node_type = "fused"
            group = snode.group
        else:
            raise RuntimeError("Unknown node type")
        node_func = func_dict[node_type]
        fx_node = graph.call_function(node_func, args=(), kwargs=None)

        def in_output(snode):
            if isinstance(snode, FusedSchedulerNode):
                return any(in_output(x) for x in snode.snodes)
            return any(isinstance(user.node, OutputNode) for user in snode.users)

        if in_output(snode):
            outputs.append(fx_node)
        name = snode.get_name()
        fx_node.name = name

        fx_node.meta["fusion_meta"] = FusionMeta(group, snode, node_type)

        if isinstance(snode, FusedSchedulerNode):
            for x in snode.snodes:
                buf_to_fx_node[x.get_name()] = fx_node
        buf_to_fx_node[name] = fx_node

        if first_node is None:
            first_node = fx_node

    # create edges between nodes
    for snode in snodes:
        name = snode.get_name()
        deps = snode.read_writes.reads

        fx_node = buf_to_fx_node[name]
        new_args = []
        for dep in deps:
            if dep.name in buf_to_fx_node:
                dep_node = buf_to_fx_node[dep.name]
            else:
                with graph.inserting_before(first_node):
                    dep_node = graph.placeholder(dep.name)
                    buf_to_fx_node[dep.name] = dep_node
            new_args.append(dep_node)

        fx_node.args = tuple(new_args)

    graph.output(outputs[0] if len(outputs) == 1 else tuple(outputs))
    return graph


@contextlib.contextmanager
def enable_aot_logging():
    compile_debug = os.environ.get("TORCH_COMPILE_DEBUG", "0") == "1"

    import torch._functorch.aot_autograd

    log = logging.getLogger(torch._functorch.aot_autograd.__name__)

    stack = contextlib.ExitStack()
    if not compile_debug:
        try:
            yield
        finally:
            stack.close()
        return

    # Enable all graphs to be logged to a file by setting the flags to True
    # and the log level of the file logger to DEBUG
    stack.enter_context(patch("functorch.compile.config.debug_partitioner", True))

    path = os.path.join(get_debug_dir(), "torchinductor")
    if not os.path.exists(path):
        os.makedirs(path)

    fh = logging.FileHandler(
        os.path.join(
            path,
            f"aot_{get_aot_graph_name()}_debug.log",
        )
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter("[%(filename)s:%(lineno)d %(levelname)s] %(message)s")
    )
    log.addHandler(fh)
    try:
        yield
    finally:
        log.removeHandler(fh)
        stack.close()


class DebugContext:
    _counter = itertools.count()

    @staticmethod
    def wrap(fn):
        @functools.wraps(fn)
        def inner(*args, **kwargs):
            with DebugContext():
                return fn(*args, **kwargs)

        return wrap_compiler_debug(inner, compiler_name="inductor")

    @staticmethod
    def create_debug_dir(folder_name):
        for n in DebugContext._counter:
            dirname = os.path.join(
                get_debug_dir(),
                "torchinductor",
                f"{folder_name}.{n}",
            )
            if not os.path.exists(dirname):
                os.makedirs(dirname)
                return dirname

    def __init__(self):
        self._prof = None
        self._path = None
        self._stack = contextlib.ExitStack()

    def copy(self, new_path: str):
        if not self._path:
            return
        assert new_path.endswith(".debug"), new_path
        if os.path.exists(new_path):
            shutil.rmtree(new_path)
        try:
            shutil.copytree(self._path, new_path)
            self._path = new_path
        except OSError:
            log.warning(
                "Failed to copy debug files from %s to %s", self._path, new_path
            )
            pass

    def fopen(self, filename):
        assert self._path
        return open(os.path.join(self._path, filename), "w")

    def filename(self, suffix):
        return os.path.join(self._path, suffix)

    def upload_tar(self):
        if config.trace.upload_tar is not None:
            import tarfile

            assert self._path
            tar_file = os.path.join(
                self._path, f"{os.path.basename(self._path)}.tar.gz"
            )
            with tarfile.open(tar_file, "w:gz") as tar:
                tar.add(self._path, arcname=os.path.basename(self._path))
            config.trace.upload_tar(tar_file)

    def __enter__(self):
        if config.debug:
            log = logging.getLogger("torch._dynamo")
            prev_level = log.level
            log.setLevel(logging.DEBUG)

            def reset_log_level(level):
                log.setLevel(level)

            self._stack.callback(reset_log_level, prev_level)

        self._stack.enter_context(V.set_debug_handler(self))

        if not config.trace.enabled:
            return

        self._path = self.create_debug_dir(get_aot_graph_name())

        if config.trace.debug_log:
            self._setup_log_capture("debug.log", logging.DEBUG)
        if config.trace.info_log:
            self._setup_log_capture("info.log", logging.INFO)
        if config.trace.compile_profile:
            self._prof = cProfile.Profile()
            self._prof.enable()

    def _setup_log_capture(self, filename, level):
        log = logging.getLogger("torch._inductor")
        fd = self._stack.enter_context(self.fopen(filename))
        ch = logging.StreamHandler(fd)
        ch.setLevel(level)
        ch.setFormatter(
            logging.Formatter("[%(filename)s:%(lineno)d %(levelname)s] %(message)s")
        )
        log.addHandler(ch)
        log.setLevel(min(log.level, level))
        self._stack.callback(log.removeHandler, ch)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._prof:
            self._prof.disable()
            self._save_profile_data()

        if self._path:
            self.upload_tar()
            log.warning("%s debug trace: %s", get_graph_being_compiled(), self._path)
        self._stack.close()

    def _save_profile_data(self):
        self._prof.dump_stats(self.filename("compile.prof"))
        with self.fopen("compile.stats") as fd:
            stats = pstats.Stats(self._prof, stream=fd)
            stats.strip_dirs()
            stats.sort_stats("cumtime")
            stats.print_stats(100)
            stats.sort_stats("tottime")
            stats.print_stats(100)

    def __getattr__(self, name):
        if config.trace.enabled and getattr(config.trace, name):
            try:
                return getattr(DebugFormatter(self), name)
            except Exception:
                log.warning("Ignoring exception in debug code", exc_info=True)
        else:

            def ignored(*args, **kwargs):
                pass

            return ignored


SchedulerNodeList = List[Any]


class DebugFormatter:
    def __init__(self, handler):
        self.fopen = handler.fopen
        self.filename = handler.filename
        self.handler = handler

    def fx_graph(self, gm: torch.fx.GraphModule, inputs: List[torch.Tensor]):
        with self.fopen("fx_graph_runnable.py") as fd:
            save_graph_repro(fd, gm, inputs, "inductor")

        with self.fopen("fx_graph_readable.py") as fd:
            fd.write(gm.print_readable(print_output=False))

    def fx_graph_transformed(
        self, gm: torch.fx.GraphModule, inputs: List[torch.Tensor]
    ):
        with self.fopen("fx_graph_transformed.py") as fd:
            fd.write(gm.print_readable(print_output=False))

    def ir_pre_fusion(self, nodes: SchedulerNodeList):
        self._write_ir("ir_pre_fusion.txt", nodes)

    def ir_post_fusion(self, nodes: SchedulerNodeList):
        self._write_ir("ir_post_fusion.txt", nodes)

    def _write_ir(self, filename: str, nodes: SchedulerNodeList):
        with self.fopen(filename) as fd:
            for node in nodes:
                fd.write(node.debug_str())
                fd.write("\n\n\n")

    def graph_diagram(self, nodes: SchedulerNodeList):
        draw_buffers(nodes, fname=self.filename("graph_diagram.svg"))

    def output_code(self, filename):
        shutil.copy(filename, self.filename("output_code.py"))
