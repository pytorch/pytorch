import collections
import contextlib
import copy
import dataclasses
import functools
import io
import itertools
import json
import logging
import os
import os.path
import pickle
import pstats
import shutil
import traceback
from collections.abc import Iterator, Sequence
from typing import Any, Callable, IO, Optional, Union
from unittest.mock import patch

import torch
from functorch.compile import draw_graph, get_aot_graph_name, get_graph_being_compiled
from torch import fx as fx
from torch._dynamo.repro.after_aot import save_graph_repro
from torch._dynamo.utils import get_debug_dir
from torch._logging import getArtifactLogger
from torch._logging._internal import trace_structured_artifact
from torch.fx.graph_module import GraphModule
from torch.fx.passes.shape_prop import _extract_tensor_metadata, TensorMetadata
from torch.fx.passes.tools_common import legalize_graph
from torch.types import FileLike
from torch.utils._ordered_set import OrderedSet
from torch.utils._pytree import tree_map

from . import config, ir  # noqa: F811, this is needed
from .ir import ExternKernelOut
from .scheduler import (
    BaseSchedulerNode,
    FusedSchedulerNode,
    NopKernelSchedulerNode,
    OutputNode,
    SchedulerNode,
    Scheduler,
)
from .virtualized import V


log = logging.getLogger(__name__)

ir_pre_fusion_log = getArtifactLogger(__name__, "ir_pre_fusion")
ir_post_fusion_log = getArtifactLogger(__name__, "ir_post_fusion")
collective_schedule_log = getArtifactLogger(__name__, "collective_schedule")
SchedulerNodeList = list[Any]
BufMeta = collections.namedtuple("BufMeta", ["name", "n_origin"])
GRAPHVIZ_COMMAND_SCALABLE = ["dot", "-Gnslimit=2", "-Gnslimit1=2", "-Gmaxiter=5000"]


@functools.cache
def has_dot() -> bool:
    return shutil.which("dot") is not None


def draw_buffers(
    nodes: list[BaseSchedulerNode],
    print_graph: bool = False,
    fname: Optional[str] = None,
) -> None:
    """
    Draw a graph in fname.svg.
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
            if isinstance(group[1], int):
                group = (group[1],)
            else:
                group = group[1]

        # gather meta data
        dtype = None
        if isinstance(node, ir.ComputedBuffer):
            dtype = node.data.dtype

        metadata = TensorMetadata(group, dtype, None, None, None, None, None)  # type: ignore[arg-type]
        node.meta["tensor_meta"] = metadata

    if print_graph:
        print(graph)

    gm = GraphModule({}, graph)
    legalize_graph(gm)
    gm.graph.lint()
    draw_graph(
        gm, fname, clear_meta=False, dot_graph_shape=config.trace.dot_graph_shape
    )


def create_fx_from_snodes(snodes: list[BaseSchedulerNode]) -> fx.Graph:
    """
    Creates a FX Graph from a list of SchedulerNode objects.
    """

    def get_fake_func(name: str) -> Callable[..., int]:
        def func1(*args: Any) -> int:
            return 0

        func1.__name__ = name
        return func1

    FusionMeta = collections.namedtuple("FusionMeta", ["group", "snode", "type"])

    buf_to_fx_node = {}
    node_to_fx_node = {}
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

        fused_name = torch._inductor.utils.get_fused_kernel_name(
            snode.get_nodes(), "original_aten"
        )
        func_name = f"{node_type}: {fused_name}"
        node_func = get_fake_func(func_name)
        kwargs = {}
        if hasattr(snode, "get_device"):
            kwargs = {"device": snode.get_device()}
        fx_node = graph.call_function(node_func, args=(), kwargs=kwargs)  # type: ignore[arg-type]

        def in_output(snode: Union[BaseSchedulerNode, FusedSchedulerNode]) -> bool:
            if isinstance(snode, FusedSchedulerNode):
                return any(in_output(x) for x in snode.snodes)
            return any(
                isinstance(user.node, OutputNode)
                for buf in snode.get_outputs()
                for user in buf.users
            )

        if in_output(snode):
            outputs.append(fx_node)
        name = snode.get_name()
        fx_node.name = name

        fx_node.meta["fusion_meta"] = FusionMeta(group, snode, node_type)

        node_to_fx_node[name] = fx_node
        for buf in snode.get_outputs():
            buf_to_fx_node[buf.get_name()] = fx_node

        if first_node is None:
            first_node = fx_node

    # create edges between nodes
    for snode in snodes:
        name = snode.get_name()
        deps = snode.read_writes.reads

        fx_node = node_to_fx_node[name]
        new_args = []
        for dep in deps:
            if dep.name in buf_to_fx_node:
                dep_node = buf_to_fx_node[dep.name]
            else:
                with graph.inserting_before(first_node):
                    dep_node = graph.placeholder(dep.name)
                    buf_to_fx_node[dep.name] = dep_node
            if dep_node == fx_node:  # to avoid cycles
                continue
            new_args.append(dep_node)

        fx_node.args = tuple(new_args)

    graph.output(outputs[0] if len(outputs) == 1 else tuple(outputs))
    return graph


def update_orig_fx_node_name_to_buf_name(
    nodes: Optional[SchedulerNodeList],
    node_name_to_buf_name: dict[str, str],
    parent_buf_name: Optional[str] = None,
    n_origins: int = 0,
) -> None:
    if nodes is None:
        return
    for node in nodes:
        # for FusedSchedulerNode, traverse recursively into get_nodes()
        buf_name = node.get_name()
        children_nodes = node.get_nodes()
        if children_nodes is not None and len(children_nodes) > 1:
            update_orig_fx_node_name_to_buf_name(
                children_nodes,
                node_name_to_buf_name,
                buf_name if parent_buf_name is None else parent_buf_name,
            )
            continue
        else:
            assert len(children_nodes) == 1 and children_nodes[0] == node

        ir_node = node.node
        if ir_node is None or ir_node.origins is None:
            continue
        for origin in ir_node.origins:
            node_name = origin.name
            # when buf1 and buf2 both have origin=node1
            # we draw node1 according to buf1
            if node_name not in node_name_to_buf_name:
                node_name_to_buf_name[node_name] = (
                    buf_name if parent_buf_name is None else parent_buf_name
                )


def get_node_name_to_buf_meta(
    node_name_to_buf_name: dict[str, str],
) -> dict[str, BufMeta]:
    buf_name_to_n_node = {}
    for node_name, buf_name in node_name_to_buf_name.items():
        if buf_name not in buf_name_to_n_node:
            buf_name_to_n_node[buf_name] = OrderedSet([node_name])
        else:
            buf_name_to_n_node[buf_name].add(node_name)

    node_name_to_buf_meta = {}
    for node_name, buf_name in node_name_to_buf_name.items():
        n_node = len(buf_name_to_n_node[buf_name])
        node_name_to_buf_meta[node_name] = BufMeta(buf_name, n_node)
    return node_name_to_buf_meta


def annotate_orig_fx_with_snodes(
    gm: torch.fx.GraphModule,
    snodes: SchedulerNodeList,
) -> None:
    """
    Creates a FX Graph from a list of SchedulerNode objects.
    """
    node_name_to_buf_name: dict[str, str] = {}
    update_orig_fx_node_name_to_buf_name(snodes, node_name_to_buf_name)
    if node_name_to_buf_name is None:
        return
    node_name_to_buf_meta = get_node_name_to_buf_meta(node_name_to_buf_name)
    for node in gm.graph.nodes:
        if node.name in node_name_to_buf_meta:
            node.meta["buf_meta"] = node_name_to_buf_meta.get(node.name)


@contextlib.contextmanager
def enable_aot_logging() -> Iterator[None]:
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
    os.makedirs(path, exist_ok=True)

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


# Used for provenance tracking
# They are not stored in DebugContext because they are not set in
# _inductor_triton_kernel_to_post_grad_node_info's Debug Context
_inductor_post_to_pre_grad_nodes: dict[str, Any] = {}
_inductor_triton_kernel_to_post_grad_node_info: dict[str, Any] = {}
_pre_grad_graph_id: Optional[int] = None
_inductor_pre_grad_node_stack_trace: dict[str, str] = {}


@contextlib.contextmanager
def reset_provenance_globals() -> Iterator[None]:
    """Context manager that resets provenance tracking globals upon entering
    and restores their original values when exiting."""
    global _pre_grad_graph_id
    global _inductor_post_to_pre_grad_nodes
    global _inductor_triton_kernel_to_post_grad_node_info

    # Store original values
    original_pre_grad_graph_id = _pre_grad_graph_id
    original_post_to_pre_grad_nodes = _inductor_post_to_pre_grad_nodes.copy()
    original_triton_kernel_to_post_grad_node_info = (
        _inductor_triton_kernel_to_post_grad_node_info.copy()
    )

    # Reset to default values
    _pre_grad_graph_id = -1
    _inductor_post_to_pre_grad_nodes = {}
    _inductor_triton_kernel_to_post_grad_node_info = {}

    try:
        yield
    finally:
        # Restore original values
        _pre_grad_graph_id = original_pre_grad_graph_id
        _inductor_post_to_pre_grad_nodes = original_post_to_pre_grad_nodes
        _inductor_triton_kernel_to_post_grad_node_info = (
            original_triton_kernel_to_post_grad_node_info
        )


class DebugContext:
    _counter = itertools.count()

    @staticmethod
    def create_debug_dir(folder_name: str) -> Optional[str]:
        debug_dir = config.trace.debug_dir or get_debug_dir()
        for n in DebugContext._counter:
            dirname = os.path.join(
                debug_dir,
                "torchinductor",
                f"{folder_name}.{n}",
            )
            if not os.path.exists(dirname):
                os.makedirs(dirname)
                return dirname
        return None

    def __init__(self) -> None:
        self._prof = None
        self._path = None
        self._stack = contextlib.ExitStack()

    def copy(self, new_path: str) -> None:
        if not self._path:
            return
        assert new_path.endswith(".debug"), new_path
        from filelock import FileLock

        try:
            with FileLock(f"{new_path}.lock"):
                if os.path.exists(new_path):
                    shutil.rmtree(new_path)
                shutil.copytree(self._path, new_path)
        except OSError:
            log.warning(
                "Failed to copy debug files from %s to %s", self._path, new_path
            )

    def fopen(
        self,
        filename: str,
        write_mode: str = "w",
        *args: Any,
        **kwargs: Any,
    ) -> IO[Any]:
        assert self._path
        return open(os.path.join(self._path, filename), write_mode, *args, **kwargs)

    @contextlib.contextmanager
    def fopen_context(
        self,
        filename: str,
        write_mode: str = "w",
        *args: Any,
        **kwargs: Any,
    ) -> Iterator[IO[Any]]:
        assert self._path
        with open(os.path.join(self._path, filename), write_mode, *args, **kwargs) as f:
            yield f

    def filename(self, suffix: str) -> str:
        assert self._path
        return os.path.join(self._path, suffix)

    def upload_tar(self) -> None:
        if config.trace.upload_tar is not None:
            import tarfile

            assert self._path
            tar_file = os.path.join(
                self._path, f"{os.path.basename(self._path)}.tar.gz"
            )
            with tarfile.open(tar_file, "w:gz") as tar:
                tar.add(self._path, arcname=os.path.basename(self._path))
            config.trace.upload_tar(tar_file)

    def __enter__(self) -> None:
        if config.debug:
            log = logging.getLogger("torch._dynamo")
            prev_level = log.level
            log.setLevel(logging.DEBUG)

            def reset_log_level(level: Any) -> None:
                log.setLevel(level)

            self._stack.callback(reset_log_level, prev_level)

        self._stack.enter_context(V.set_debug_handler(self))

        if not config.trace.enabled:
            return

        self._path = self.create_debug_dir(get_aot_graph_name())  # type: ignore[assignment]

        if config.trace.debug_log:
            self._setup_log_capture("debug.log", logging.DEBUG)
        if config.trace.info_log:
            self._setup_log_capture("info.log", logging.INFO)

    def _setup_log_capture(
        self,
        filename: str,
        level: int,
    ) -> None:
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

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        if self._prof:
            self._prof.disable()
            self._save_profile_data()

        if self._path:
            self.upload_tar()
            log.warning("%s debug trace: %s", get_graph_being_compiled(), self._path)
        self._stack.close()

    def _save_profile_data(self) -> None:
        assert self._prof
        self._prof.dump_stats(self.filename("compile.prof"))
        with self.fopen("compile.stats") as fd:
            stats = pstats.Stats(self._prof, stream=fd)
            stats.strip_dirs()
            stats.sort_stats("cumtime")
            stats.print_stats(100)
            stats.sort_stats("tottime")
            stats.print_stats(100)

    def __getattr__(self, name: str) -> Optional[Callable[..., None]]:
        if config.trace.enabled and getattr(config.trace, name):
            try:
                return getattr(DebugFormatter(self), name)
            except Exception:
                log.warning("Ignoring exception in debug code", exc_info=True)
                return None
        else:

            def ignored(*args: Any, **kwargs: Any) -> None:
                pass

            return ignored


class DebugFormatter:
    def __init__(self, handler: DebugContext) -> None:
        self.fopen = handler.fopen
        self.fopen_context = handler.fopen_context
        self.filename = handler.filename
        self.handler = handler

    def fx_graph(
        self,
        gm: torch.fx.GraphModule,
        inputs: list[torch.Tensor],
    ) -> None:
        with self.fopen("fx_graph_runnable.py") as fd:
            save_dir = None
            if torch._inductor.config.trace.save_real_tensors:
                inputs = torch._subclasses.fake_utils.try_convert_fake_to_real(inputs)
                save_dir = os.path.dirname(fd.name)

            # dont try to use stable hash torchinductor compilation if saving real tensors
            # and avoid recursively trying to save real tensors inside of the inductor compilation
            # regardless
            stable_hash = torch._inductor.config.trace.save_real_tensors
            with torch._inductor.config.patch(
                {"trace.enabled": False, "trace.save_real_tensors": False}
            ):
                save_graph_repro(
                    fd,
                    gm,
                    inputs,
                    "inductor",
                    save_dir=save_dir,
                    stable_hash=stable_hash,
                )

        with self.fopen("fx_graph_readable.py") as fd:
            fd.write(gm.print_readable(print_output=False))

    def fx_graph_transformed(
        self,
        gm: torch.fx.GraphModule,
        inputs: list[torch.Tensor],
    ) -> None:
        with self.fopen("fx_graph_transformed.py") as fd:
            fd.write(gm.print_readable(print_output=False))

    def ir_pre_fusion(self, nodes: SchedulerNodeList) -> None:
        with self.fopen("ir_pre_fusion.txt") as fd:
            fd.write(self._write_ir(nodes))

    def ir_post_fusion(self, nodes: SchedulerNodeList) -> None:
        with self.fopen("ir_post_fusion.txt") as fd:
            fd.write(self._write_ir(nodes))

    @staticmethod
    def _write_ir(nodes: SchedulerNodeList) -> str:
        buf = io.StringIO()
        for node in nodes:
            buf.write(node.debug_str())
            buf.write("\n\n\n")
        return buf.getvalue()

    def graph_diagram(self, nodes: SchedulerNodeList) -> None:
        draw_buffers(nodes, fname=self.filename("graph_diagram.svg"))

    def draw_orig_fx_graph(
        self,
        gm: torch.fx.GraphModule,
        nodes: SchedulerNodeList,
    ) -> None:
        annotate_orig_fx_with_snodes(gm, nodes)
        draw_graph(
            gm,
            fname=self.filename("orig_fx_graph_diagram.svg"),
            clear_meta=False,
            prog=GRAPHVIZ_COMMAND_SCALABLE,
            parse_stack_trace=True,
            dot_graph_shape=config.trace.dot_graph_shape,
        )

    def output_code(self, filename: str, extension: str = "py") -> None:
        shutil.copy(filename, self.filename(f"output_code.{extension}"))

    def log_autotuning_results(
        self,
        name: str,
        input_nodes: list[ir.IRNode],
        timings: dict["ChoiceCaller", float],  # type: ignore[name-defined] # noqa: F821
        elapse: float,
        precompile_elapse: float,
        prescreening_elapse: Optional[float],
    ) -> None:
        from .ir import FixedLayout

        def build_node_info(node: ir.IRNode) -> dict[str, str]:
            if hasattr(node, "name"):
                node_name = node.name
            else:
                node_name = ""
            node_info = {
                "name": node_name,
                "type": type(node).__name__,
            }
            try:
                layout = node.get_output_spec()
                if isinstance(layout, FixedLayout):
                    offset = 0
                    try:
                        offset = int(layout.offset)
                    except Exception:
                        try:
                            offset = V.graph.sizevars.size_hint(
                                layout.offset, fallback=0
                            )
                        except Exception:
                            pass
                    static_layout = FixedLayout(
                        layout.device,
                        dtype=layout.dtype,
                        size=[*V.graph.sizevars.size_hints(layout.size)],
                        stride=[*V.graph.sizevars.size_hints(layout.stride)],
                        offset=offset,
                    )
                    node_info["layout"] = str(static_layout)
                else:
                    node_info["layout"] = str(layout)
            except Exception:
                pass
            try:
                node_info["dtype"] = str(node.get_dtype())
            except Exception:
                pass
            try:
                node_info["device"] = str(node.get_device())
            except Exception:
                pass
            try:
                node_info["stride"] = str(
                    V.graph.sizevars.size_hints(node.get_stride())
                )
            except Exception:
                pass
            try:
                node_info["size"] = str(V.graph.sizevars.size_hints(node.get_size()))  # type: ignore[arg-type]
            except Exception:
                pass
            try:
                node_info["numel"] = str(V.graph.sizevars.size_hint(node.get_numel()))
            except Exception:
                pass
            if hasattr(node, "data") and isinstance(node.data, ir.IRNode):
                node_info["data"] = build_node_info(node.data)
            return node_info

        general_properties = {
            "op_name": name,
            "cuda_device_name": torch.cuda.get_device_name(),
            "cuda_device_count": torch.cuda.device_count(),
            "input_nodes": [build_node_info(node) for node in input_nodes],
            "autotuning_time": elapse,
            "precompile_time": precompile_elapse,
            "prescreening_time": prescreening_elapse,
        }
        with self.fopen_context(
            "autotuning_result_json_list.txt", "at", encoding="utf-8"
        ) as fd:
            for caller, time in timings.items():
                info_dict = dict(caller.info_dict())
                info_dict.update(general_properties)
                info_dict["benchmark_result"] = time
                json.dump(info_dict, fd)
                fd.write("\n")

    def dump_collective_schedule(self, schedule: list[dict[str, Any]]) -> None:
        with self.fopen("collective_schedule.json", "w", encoding="utf-8") as fd:
            json.dump(schedule, fd)
        try:
            trace_structured_artifact(
                "inductor_collective_schedule",
                "string",  # encoding
                payload_fn=lambda: json.dumps(schedule, separators=(",", ":")),
            )
        except Exception:
            log.debug("Failed to log inductor_collective_schedule via structured logging", exc_info=True)


def log_ir_pre_fusion(nodes: SchedulerNodeList) -> None:
    if ir_pre_fusion_log.isEnabledFor(logging.INFO):
        ir_pre_fusion_log.info("BEFORE FUSION\n%s", DebugFormatter._write_ir(nodes))

    V.debug.ir_pre_fusion(nodes)


def log_ir_post_fusion(nodes: SchedulerNodeList) -> None:
    if ir_post_fusion_log.isEnabledFor(logging.INFO):
        ir_post_fusion_log.info("AFTER FUSION\n%s", DebugFormatter._write_ir(nodes))

    V.debug.ir_post_fusion(nodes)


def log_collective_schedule(scheduler: "Scheduler") -> None:
    schedule = [
        {
            "op_name": getattr(node.node, "python_kernel_name", None)
                      or node.node.kernel
        }
        for node in scheduler.nodes
        if isinstance(getattr(node, "node", None), ir._CollectiveKernel)
    ]

    V.debug.dump_collective_schedule(schedule)


@dataclasses.dataclass
class TensorMetadataHolder:
    tensor_metadata: TensorMetadata
    device: torch.device


save_args_cnt = itertools.count()


def create_mapping_pre_post_grad_nodes(
    pre_grad_graph_id: Optional[int],
    post_to_pre_grad_nodes_json: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """
    Create bidirectional mappings between pre_grad graph nodes
    and post_grad graph code nodes, and vice versa.
    """
    # return a dummy dict if there's any error
    empty_return: dict[str, dict[str, Any]] = {
        "preToPost": {},
        "postToPre": {},
    }

    log.info("Creating node mappings for provenance tracking")

    if not isinstance(post_to_pre_grad_nodes_json, dict):
        log.error("Provenance tacking error: post_to_pre_grad_nodes_json is not a dict")
        return empty_return

    if not isinstance(pre_grad_graph_id, int):
        log.error("Provenance tacking error: pre_grad_graph_id is not an int")
        return empty_return

    pre_to_post: dict[str, Any] = collections.defaultdict(OrderedSet)
    post_to_pre: dict[str, Any] = collections.defaultdict(OrderedSet)

    try:

        def check_format(node: dict[str, Any]) -> bool:
            if not isinstance(node, dict):
                log.error(
                    "Provenance tacking error: node provenance in post_to_pre_grad_nodes_json is not a dict"
                )
                return False
            if "graph_id" not in node or "name" not in node or "from_node" not in node:
                log.error(
                    "Provenance tacking error: node provenance in post_to_pre_grad_nodes_json has wrong format"
                )
                return False
            return True

        for outer_key, node_array in post_to_pre_grad_nodes_json.items():
            if not isinstance(node_array, list):
                log.error(
                    "Provenance tacking error: post_to_pre_grad_nodes_json value is not a list"
                )
                return empty_return
            for node in node_array:
                if not check_format(node):
                    return empty_return
                # Check the current node first
                if node.get("graph_id") == pre_grad_graph_id:
                    pre_to_post[node["name"]].add(outer_key)
                    post_to_pre[outer_key].add(node["name"])

                # Check nested from_node array recursively, add node with the right graph_id to the map
                stack = [(n, outer_key) for n in node.get("from_node", [])]
                while stack:
                    current_node, parent_key = stack.pop()
                    if not check_format(current_node):
                        return empty_return
                    if current_node.get("graph_id") == pre_grad_graph_id:
                        pre_to_post[current_node["name"]].add(parent_key)
                        post_to_pre[parent_key].add(current_node["name"])
                    stack.extend(
                        (n, parent_key) for n in current_node.get("from_node", [])
                    )

        def convert_sets_to_lists(d: dict[str, Any]) -> None:
            for key in d:
                d[key] = list(d[key])
            d = dict(d)

        # convert to list because set is not JSON serializable
        convert_sets_to_lists(pre_to_post)
        convert_sets_to_lists(post_to_pre)
        return {
            "preToPost": pre_to_post,
            "postToPre": post_to_pre,
        }
    except Exception as e:
        # Since this is just logging code, it should never interfere with regular
        # program execution, so we use this try-except to guard against any error
        log.error("Unexpected error in create_node_mapping: %s", e)
        log.error("post_to_pre_grad_nodes_json:  %s", post_to_pre_grad_nodes_json)
        log.error("pre_grad_graph_id:  %s", pre_grad_graph_id)
        log.error(traceback.format_exc())
        return empty_return


def create_node_mapping_kernel_to_post_grad(
    triton_kernel_to_post_grad_json: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Create bidirectional mappings between triton kernel name and post_grad
    graph code nodes, and vice versa.
    """

    # return a dummy dict if there's any error
    empty_return: dict[str, dict[str, Any]] = {
        "cppCodeToPost": {},
        "postToCppCode": {},
    }

    log.info("Creating node mappings for provenance tracking")

    if not isinstance(triton_kernel_to_post_grad_json, dict):
        log.error(
            "Provenance tacking error: triton_kernel_to_post_grad_json is not a dict"
        )
        return empty_return

    post_to_cpp_code: dict[str, Any] = collections.defaultdict(OrderedSet)

    try:
        for outer_key, node_array in triton_kernel_to_post_grad_json.items():
            if not isinstance(node_array, list):
                log.error(
                    "Provenance tacking error: triton_kernel_to_post_grad_json value is not a list"
                )
                return empty_return
            for curr_node in node_array:
                post_to_cpp_code[curr_node].add(outer_key)

        def convert_sets_to_lists(d: dict[str, Any]) -> None:
            for key in d:
                d[key] = list(d[key])
            d = dict(d)

        # convert to list because set is not JSON serializable
        convert_sets_to_lists(post_to_cpp_code)
        return {
            "cppCodeToPost": triton_kernel_to_post_grad_json,
            "postToCppCode": post_to_cpp_code,
        }
    except Exception as e:
        # Since this is just logging code, it should never interfere with regular
        # program execution, so we use this try-except to guard against any error
        log.error("Unexpected error in create_node_mapping: %s", e)
        log.error(
            "triton_kernel_to_post_grad_json:  %s", triton_kernel_to_post_grad_json
        )
        log.error(traceback.format_exc())
        return empty_return

def dump_inductor_provenance_info(
    filename: str = "inductor_generated_kernel_to_post_grad_nodes.json",
) -> dict[str, Any]:
    global _pre_grad_graph_id
    global _inductor_post_to_pre_grad_nodes
    global _inductor_triton_kernel_to_post_grad_node_info
    if config.trace.enabled:
        with V.debug.fopen(filename, "w") as fd:
            log.info("Writing provenance tracing debugging info to %s", fd.name)
            json.dump(_inductor_triton_kernel_to_post_grad_node_info, fd)
    node_mapping = {}
    if _pre_grad_graph_id:
        node_mapping_kernel = create_node_mapping_kernel_to_post_grad(
            _inductor_triton_kernel_to_post_grad_node_info
        )
        node_mapping = {
            **_inductor_post_to_pre_grad_nodes,
            **node_mapping_kernel,
        }
        if config.trace.enabled:
            with V.debug.fopen(
                "inductor_provenance_tracking_node_mappings.json", "w"
            ) as fd:
                json.dump(node_mapping, fd)
    return node_mapping


def set_kernel_post_grad_provenance_tracing(
    node_schedule: Union[Sequence[BaseSchedulerNode], ExternKernelOut],
    kernel_name: str,
    is_extern: bool = False,
) -> None:
    from .codegen.simd_kernel_features import DisableReduction, EnableReduction

    global _inductor_triton_kernel_to_post_grad_node_info
    if is_extern:
        assert isinstance(node_schedule, ExternKernelOut)
        curr_node_info = _inductor_triton_kernel_to_post_grad_node_info.setdefault(
            kernel_name, []
        )
        # 'origins' on IR nodes gives what FX IR nodes contributed to any given fused kernel.
        # "origin_node" is more precise and says that the contents of this node corresponds
        # EXACTLY to the output of a particular FX node, but it's not always available
        if node_schedule.origin_node:
            origin_node_name = node_schedule.origin_node.name
            if origin_node_name not in curr_node_info:
                curr_node_info.append(origin_node_name)
        else:
            curr_node_info.extend(
                origin.name
                for origin in node_schedule.origins
                if origin.name not in curr_node_info
            )
    else:
        assert isinstance(node_schedule, list)
        for snode in node_schedule:
            if snode not in (EnableReduction, DisableReduction):
                if snode.node is not None:
                    curr_node_info = (
                        _inductor_triton_kernel_to_post_grad_node_info.setdefault(
                            kernel_name, []
                        )
                    )
                    curr_node_info.extend(
                        origin.name
                        for origin in snode.node.origins
                        if origin.name not in curr_node_info
                    )


def save_args_for_compile_fx_inner(*args: Any, **kwargs: Any) -> None:
    """
    This function is used to save arguments for a compile_fx_inner function call
    to the file system.  Later on one can replay the compile_fx_inner call
    with the saved arguments using load_args_and_run_compile_fx_inner.
    """

    folder = "/tmp/inductor_saved_args"
    if not os.path.exists(folder):
        os.mkdir(folder)

    def handle_tensor(x: Any) -> Any:
        """
        Pickle FakeTensor will result in error:
        AttributeError: Can't pickle local object 'WeakValueDictionary.__init__.<locals>.remove'

        Convert all Tensor to metadata. This may also makes pickle faster.
        """
        if isinstance(x, torch.Tensor):
            return TensorMetadataHolder(_extract_tensor_metadata(x), x.device)
        else:
            return x

    args_to_save, kwargs_to_save = tree_map(handle_tensor, (args, kwargs))

    fn_name = "compile_fx_inner"
    path = f"{folder}/{fn_name}_{next(save_args_cnt)}.pkl"
    with open(path, "wb") as f:
        pickle.dump((args_to_save, kwargs_to_save), f)

    if log.isEnabledFor(logging.DEBUG):
        message = f"""
Arguments for a compile_fx_inner call is saved to {path}. To replay the call,
run the following:

from torch._inductor.debug import load_args_and_run_compile_fx_inner
load_args_and_run_compile_fx_inner({path!r})
        """
        # call print rather than log.debug. log.debug will print message
        # prefix for each line which makes the code snippet harder to be
        # copied.
        # Not a big deal since the code is already been guarded by checking
        # the log level.
        print(message)


def load_args_and_run_compile_fx_inner(path: str) -> Any:
    from torch._inductor.compile_fx import compile_fx_inner

    with open(path, "rb") as f:
        args, kwargs = pickle.load(f)

    def handle_tensor(x: Any) -> Any:
        if isinstance(x, TensorMetadataHolder):
            return torch._dynamo.testing.rand_strided(
                x.tensor_metadata.shape,
                x.tensor_metadata.stride,
                x.tensor_metadata.dtype,
                x.device,
            )
        else:
            return x

    fake_mode = torch._subclasses.FakeTensorMode(allow_non_fake_inputs=True)
    with fake_mode, config.patch("save_args", False):
        args, kwargs = tree_map(handle_tensor, (args, kwargs))
        return compile_fx_inner(*args, **kwargs)


def aot_inductor_minifier_wrapper(
    func: Callable[..., str],
    exported_program: torch.export.ExportedProgram,
    *,
    inductor_configs: dict[str, Any],
    package_path: Optional[FileLike] = None,
) -> str:
    from torch._dynamo.debug_utils import AccuracyError
    from torch._dynamo.repro.aoti import dump_to_minify
    from torch._inductor import config
    from torch._inductor.compile_fx import _aoti_flatten_inputs

    use_minifier = config.aot_inductor.dump_aoti_minifier

    gm = exported_program.module()
    assert isinstance(gm, torch.fx.GraphModule)

    args, kwargs = exported_program.example_inputs

    try:
        if use_minifier and config.aot_inductor.repro_level == 3:
            # Always dump the original module in case we have segfaults
            dump_to_minify(
                exported_program,
                "aot_inductor",
                options=inductor_configs,
            )
        if use_minifier and config.aot_inductor.repro_level == 4:
            # Check for accuracy
            # We will first flatten the inputs before compiling and checking for accuracy.
            # This is ok because we will flatten the inputs in the minifier anyway.
            gm_copy = copy.deepcopy(gm)
            example_inputs_copy = copy.deepcopy(exported_program.example_inputs)
            config_copy = copy.deepcopy(inductor_configs)
            flat_example_inputs, config_copy = _aoti_flatten_inputs(
                gm_copy,
                example_inputs_copy[0],
                example_inputs_copy[1],
                options=config_copy,
            )
            tuple_inputs = tuple(flat_example_inputs)
            flattened_ep = torch.export.export(gm_copy, tuple_inputs, strict=False)
            func(
                flattened_ep.module(),
                tuple_inputs,
                inductor_configs=config_copy,
                package_path=package_path,
                load_and_run=True,
                check_accuracy="accuracy",
            )

        return func(
            gm,
            args,
            kwargs,
            inductor_configs=inductor_configs,
            package_path=package_path,
            load_and_run=use_minifier,
        )
    except AccuracyError as e:
        dump_to_minify(
            exported_program,
            "aot_inductor_accuracy",
            command="minify",
            options=inductor_configs,
        )
        log.warning("Accuracy failed")
        raise e
    except Exception as e:
        if use_minifier:
            command = "minify"

            if config.aot_inductor.repro_level == 1:
                command = "run"

            dump_to_minify(
                exported_program,
                "aot_inductor",
                command=command,
                options=inductor_configs,
            )
        raise e
