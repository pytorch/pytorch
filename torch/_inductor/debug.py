import collections
import contextlib
import dataclasses
import functools
import io
import itertools
import logging
import os
import os.path
import pickle
import pstats
import shutil
import subprocess
from typing import Any, Callable, Dict, IO, Iterator, List, Optional, Type, Union
from unittest.mock import patch

import torch
from functorch.compile import draw_graph, get_aot_graph_name, get_graph_being_compiled
from torch import fx as fx
from torch._dynamo.repro.after_aot import save_graph_repro
from torch._dynamo.utils import get_debug_dir
from torch.fx.graph_module import GraphModule
from torch.fx.passes.shape_prop import _extract_tensor_metadata, TensorMetadata
from torch.fx.passes.tools_common import legalize_graph
from torch.utils._ordered_set import OrderedSet
from torch.utils._pytree import tree_map

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

SchedulerNodeList = List[Any]
BufMeta = collections.namedtuple("BufMeta", ["name", "n_origin"])
GRAPHVIZ_COMMAND_SCALABLE = ["dot", "-Gnslimit=2", "-Gnslimit1=2", "-Gmaxiter=5000"]


@functools.lru_cache(None)
def has_dot() -> bool:
    try:
        subprocess.check_output(["which", "dot"], stderr=subprocess.PIPE)
        return True
    except subprocess.SubprocessError:
        return False


def draw_buffers(
    nodes: List[BaseSchedulerNode],
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


def create_fx_from_snodes(snodes: List[BaseSchedulerNode]) -> fx.Graph:
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
    node_name_to_buf_name: Dict[str, str],
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
    node_name_to_buf_name: Dict[str, str]
) -> Dict[str, BufMeta]:
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
    node_name_to_buf_name: Dict[str, str] = {}
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
        exc_type: Optional[Type[BaseException]],
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
        inputs: List[torch.Tensor],
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
        inputs: List[torch.Tensor],
    ) -> None:
        with self.fopen("fx_graph_transformed.py") as fd:
            fd.write(gm.print_readable(print_output=False))

    def ir_pre_fusion(self, nodes: SchedulerNodeList) -> None:
        self._write_ir("ir_pre_fusion.txt", nodes)

    def ir_post_fusion(self, nodes: SchedulerNodeList) -> None:
        self._write_ir("ir_post_fusion.txt", nodes)

    def _write_ir(
        self,
        filename: str,
        nodes: SchedulerNodeList,
    ) -> None:
        with self.fopen(filename) as fd:
            log.info("Writing debug ir to  %s", fd.name)
            for node in nodes:
                fd.write(node.debug_str())
                fd.write("\n\n\n")

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

    def output_code(self, filename: str) -> None:
        shutil.copy(filename, self.filename("output_code.py"))

    def log_autotuning_results(
        self,
        name: str,
        input_nodes: List[ir.IRNode],
        timings: Dict["ChoiceCaller", float],  # type: ignore[name-defined] # noqa: F821
        elapse: float,
        precompile_elapse: float,
    ) -> None:
        import json

        from .ir import FixedLayout

        def build_node_info(node: ir.IRNode) -> Dict[str, str]:
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
                node_info["size"] = str(V.graph.sizevars.size_hints(node.get_size()))
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


@dataclasses.dataclass
class TensorMetadataHolder:
    tensor_metadata: TensorMetadata
    device: torch.device


save_args_cnt = itertools.count()


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
    inductor_configs: Dict[str, Any],
    package_path: Optional[Union[str, io.BytesIO]] = None,
) -> str:
    from torch._inductor import config

    use_minifier = config.aot_inductor.dump_aoti_minifier

    gm = exported_program.module()
    assert isinstance(gm, torch.fx.GraphModule)

    args, kwargs = exported_program.example_inputs

    try:
        return func(
            gm,
            args,
            kwargs,
            inductor_configs=inductor_configs,
            package_path=package_path,
            load_and_run=use_minifier,
        )
    except Exception as e:
        if use_minifier:
            # TODO: check accuracy and re-direct to minifier
            from torch._dynamo.repro.aoti import dump_to_minify

            dump_to_minify(
                exported_program,
                "compile_fx_aot",
                options=inductor_configs,
            )
        raise e
