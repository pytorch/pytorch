# mypy: allow-untyped-defs
"""
This is a script for launching PyTorch inference on Intel® Xeon® Scalable Processors with optimal configurations.

Single instance inference, multi-instance inference are enabled.

Note: term "instance" here doesn't refer to a cloud instance. This script is executed as a single process. It invokes
multiple "instances" which are formed from multiple threads for each. "instance" is kind of group of threads in this
context.

Command `torch-xeon-launcher` is equivalent to `python -m torch.backends.xeon.run_cpu`.

Illustrated as below:

::

    +---------------------+----------------------+-------+
    |        process      |        thread        | core  |
    +=====================+======================+=======+
    | torch-xeon-launcher | instance 0: thread 0 |   0   |
    |                     |             thread 1 |   1   |
    |                     +----------------------+-------+
    |                     | instance 1: thread 0 |   2   |
    |                     |             thread 1 |   3   |
    |                     +----------------------+-------+
    |                     | ...                  |  ...  |
    |                     +----------------------+-------+
    |                     | instance N: thread 0 |   M   |
    |                     |             thread 1 |  M+1  |
    +---------------------+----------------------+-------+

To get the peak performance on Intel® Xeon® Scalable Processors, the script optimizes the configuration of thread and memory
management. For thread management, the script configures thread affinity and the preload of Intel OMP library.
For memory management, it configures NUMA binding and preload optimized memory allocation library (e.g. tcmalloc, jemalloc).

Environment variables that will be set by this script:

+------------------+-------------------------------------------------------------------------------------------------+
| Environ Variable |                                             Value                                               |
+==================+=================================================================================================+
|    LD_PRELOAD    | Depending on knobs you set, <lib>/libiomp5.so, <lib>/libjemalloc.so, <lib>/libtcmalloc.so might |
|                  | be appended to LD_PRELOAD.                                                                      |
+------------------+-------------------------------------------------------------------------------------------------+
|   KMP_AFFINITY   | If libiomp5.so is preloaded, KMP_AFFINITY could be set to "granularity=fine,compact,1,0".       |
+------------------+-------------------------------------------------------------------------------------------------+
|   KMP_BLOCKTIME  | If libiomp5.so is preloaded, KMP_BLOCKTIME is set to "1".                                       |
+------------------+-------------------------------------------------------------------------------------------------+
|  OMP_NUM_THREADS | value of ncores_per_instance                                                                    |
+------------------+-------------------------------------------------------------------------------------------------+
|    MALLOC_CONF   | If libjemalloc.so is preloaded, MALLOC_CONF will be set to                                      |
|                  | "oversize_threshold:1,background_thread:true,metadata_thp:auto".                                |
+------------------+-------------------------------------------------------------------------------------------------+

*Note*: This script respects environment variables set preliminarily. I.e. If you set the environment variables
mentioned above before running the script, the script will not overwrite the values in the script.

Arguments:
~~~~~~~~~~

+-------------------------+------+-----------+-----------------------------------------------------------------------+
|        Arguments        | Type |  Default  | Description                                                           |
+=========================+======+===========+=======================================================================+
| `-h`, `--help`          |  --  |    --     | Show this help message and exit                                       |
+-------------------------+------+-----------+-----------------------------------------------------------------------+
| `-m`, `--module`        |  --  |   False   | Changes each process to interpret the launch script  as a python      |
|                         |      |           | module, executing with the same behavior as 'python -m'.              |
+-------------------------+------+-----------+-----------------------------------------------------------------------+
| `--no-python`           |  --  |   False   | Avoid applying `python` to execute `program`.                         |
+-------------------------+------+-----------+-----------------------------------------------------------------------+
| `--log-dir`             | str  |    ''     | The log file directory. Setting it to empty ('') disables logging to  |
|                         |      |           | files.                                                                |
+-------------------------+------+-----------+-----------------------------------------------------------------------+
| `--log-file-prefix`     | str  |   'run'   | Log file name prefix                                                  |
+-------------------------+------+-----------+-----------------------------------------------------------------------+
| `--ninstances`          | int  |     0     | Number of instances                                                   |
+-------------------------+------+-----------+-----------------------------------------------------------------------+
|                         |      |           | Inside the multi instance list, execute specific instances at indice  |
| `--instance-idx`        | str  |    -1     | (count from 0, separate by comma(,)). If it is set to -1, run all of  |
|                         |      |           | instances. If it is set to an individual number, run the instance at  |
|                         |      |           | that specific index.                                                  |
+-------------------------+------+-----------+-----------------------------------------------------------------------+
|                         |      |           | Number of cores per instance. It has to be an integer larger than or  |
|                         |      |           | equal to `-1`. When set to `0`, cores are evenly assigned to each     |
|                         |      |           | instance. If number of cores cannot be divided by number of instances,|
| `--ncores-per-instance` | int  |     0     | residual cores are unused. When set to `-1`, cores are evenly assigned|
|                         |      |           | to each instance as much as possible to fully utilize all cores. When |
|                         |      |           | set to a number larger than `0`, designated number of cores are       |
|                         |      |           | assigned to each instance.                                            |
+-------------------------+------+-----------+-----------------------------------------------------------------------+
|                         |      |           | Specify nodes list for multiple instances to run on, in format of list|
| `--nodes-list`          | str  |    ''     | of single node ids "node_id,node_id,..." or list of node ranges       |
|                         |      |           | "node_id-node_id,...". By default all nodes will be used.             |
+-------------------------+------+-----------+-----------------------------------------------------------------------+
|                         |      |           | Specify cores list for multiple instances to run on, in format of list|
| `--cores-list`          | str  |    ''     | of single core ids "core_id,core_id,..." or list of core ranges       |
|                         |      |           | "core_id-core_id,...". By default all cores will be used.             |
+-------------------------+------+-----------+-----------------------------------------------------------------------+
| `--use-logical-cores`   |  --  |   False   | Use logical cores on the workloads or not. By default, only physical  |
|                         |      |           | cores are used.                                                       |
+-------------------------+------+-----------+-----------------------------------------------------------------------+
| `--use-e-cores`         |  --  |   False   | Use Efficient-Cores on the workloads or not. By default, only         |
|                         |      |           | Performance-Cores are used.                                           |
+-------------------------+------+-----------+-----------------------------------------------------------------------+
| `--memory-allocator`    | str  |   'auto'  | Choose which memory allocator to run the workloads with.              |
|                         |      |           | Supported choices are ['auto', 'default', 'tcmalloc', 'jemalloc'].    |
+-------------------------+------+-----------+-----------------------------------------------------------------------+
| `--omp-runtime`         | str  |   'auto'  | Choose which OpenMP runtime to run the workloads with.                |
|                         |      |           | Supported choices are ['auto', 'default', 'intel'].                   |
+-------------------------+------+-----------+-----------------------------------------------------------------------+
| `--multi-task-manager`  | str  |   'auto'  | Choose which multi task manager to run the workloads with.            |
|                         |      |           | Supported choices are ['auto', 'none', 'numactl', 'taskset'].         |
+-------------------------+------+-----------+-----------------------------------------------------------------------+
| `--bind-numa-node`      |  --  |   False   | Bind instances to be executed on cores on a single NUMA node.         |
+-------------------------+------+-----------+-----------------------------------------------------------------------+
|                         |      |           | Tell how cores are distributed over instances when only part of all   |
|                         |      |           | cores are needed on a machine with multiple NUMA nodes.               |
| `--strategy`            | str  | 'scatter' | Supported choices are ['scatter', 'close']. With 'scatter', instances |
|                         |      |           | are distributed evenly as much as possible over all available NUMA    |
|                         |      |           | nodes. While with 'close', instances are assigned to cores in order   |
|                         |      |           | continuously.                                                         |
+-------------------------+------+-----------+-----------------------------------------------------------------------+
| `--latency-mode`        |  --  |   False   | Use 4 cores per instance over all physical cores.                     |
+-------------------------+------+-----------+-----------------------------------------------------------------------+
| `--throughput-mode`     |  --  |   False   | Run one instance per node with all physical cores.                    |
+-------------------------+------+-----------+-----------------------------------------------------------------------+
|                         |      |           | Enable benchmark config. JeMalloc's MALLOC_CONF has been tuned for low|
| `--benchmark`           |  --  |   False   | latency. Recommend to use this for benchmarking purpose; for other use|
|                         |      |           | cases, this MALLOC_CONF may cause Out-of-Memory crash.                |
+-------------------------+------+-----------+-----------------------------------------------------------------------+

Usage Examples:
~~~~~~~~~~~~~~~

Single instance inference
-------------------------

1. Run single-instance inference on a single node with all CPU nodes.

::

   torch-xeon-launcher --throughput-mode script.py args

2. Run single-instance inference on a single CPU node.

::

   torch-xeon-launcher --nodes-list 1 script.py args

Multi-instance inference
------------------------

1. Multi-instance
   By default this tool runs one process per node. If you want to set the instance numbers and core per instance,
   --ninstances and  --ncores-per-instance should be set.

   .. code-block:: bash

      torch-xeon-launcher -- python_script args

   eg: on an Intel® Xeon® Scalable Processor with 14 instance, 4 cores per instance

   .. code-block:: bash

      torch-xeon-launcher --ninstances 14 --ncores-per-instance 4 python_script args

2. Run single-instance inference among multiple instances.
   By default, runs all ninstances. If you want to independently run a single instance among ninstances, specify rank.

   eg: run 0th instance on an Intel® Xeon® Scalable Processor with 2 instance (i.e., numactl -C 0-27)

   .. code-block:: bash

      torch-xeon-launcher --ninstances 2 --instance_idx 0 python_script args

   eg: run 1st instance on an Intel® Xeon® Scalable Processor with 2 instance (i.e., numactl -C 28-55)

   .. code-block:: bash

      torch-xeon-launcher --ninstances 2 --instance_idx 1 python_script args

   eg: run 0th instance on an Intel® Xeon® Scalable Processor with 2 instance, 2 cores per instance,
   first four cores (i.e., numactl -C 0-1)

   .. code-block:: bash

      torch-xeon-launcher --cores-list "0,1,2,3" --ninstances 2 --ncores-per-instance 2 --instance_idx 0 python_script args

3. To look up what optional arguments this module offers:

   .. code-block:: bash

      torch-xeon-launcher --help

Memory allocator
----------------

`--memory-allocator` can be used to enable different memory allcator.

.. code-block:: bash

   --memory-allocator default
   --memory-allocator tcmalloc
   --memory-allocator jemalloc

"""

import argparse
import glob
import logging
import os
import platform
from argparse import RawTextHelpFormatter
from datetime import datetime

from ._launcher_multi_instances import MultiInstancesLauncher


__all__ = [
    "main",
]


def _add_deprecated_params(parser):
    group = parser.add_argument_group("Deprecated Arguments")
    group.add_argument(
        "--enable-tcmalloc",
        "--enable_tcmalloc",
        action="store_true",
        default=False,
        help="Deprecated by --memory-allocator.",
    )
    group.add_argument(
        "--enable-jemalloc",
        "--enable_jemalloc",
        action="store_true",
        default=False,
        help="Deprecated by --memory-allocator.",
    )
    group.add_argument(
        "--use-default-allocator",
        "--use_default_allocator",
        action="store_true",
        default=False,
        help="Deprecated by --memory-allocator.",
    )
    group.add_argument(
        "--skip-cross-node-cores",
        "--skip_cross_node_cores",
        action="store_true",
        default=False,
        help="Deprecated by --bind-numa-node.",
    )
    group.add_argument(
        "--node_id",
        "--node-id",
        metavar="\b",
        type=int,
        default=-1,
        help="Deprecated by --nodes-list.",
    )
    group.add_argument(
        "--use_logical_core",
        "--use-logical-core",
        action="store_true",
        default=False,
        help="Deprecated by --use-logical-cores.",
    )
    group.add_argument(
        "--disable_numactl",
        "--disable-numactl",
        action="store_true",
        default=False,
        help="Deprecated by --multi-task-manager.",
    )
    group.add_argument(
        "--disable_taskset",
        "--disable-taskset",
        action="store_true",
        default=False,
        help="Deprecated by --multi-task-manager.",
    )
    group.add_argument(
        "--disable_iomp",
        "--disable-iomp",
        action="store_true",
        default=False,
        help="Deprecated by --omp-runtime.",
    )
    group.add_argument(
        "--core_list",
        "--core-list",
        metavar="\b",
        type=str,
        default="",
        help="Deprecated by --cores-list.",
    )
    group.add_argument(
        "--log_path",
        "--log-path",
        type=str,
        default="",
        help="Deprecated by --log-dir.",
    )
    group.add_argument(
        "--multi_instance",
        "--multi-instance",
        action="store_true",
        default=False,
        help="Deprecated. Will be removed.",
    )
    group.add_argument(
        "--rank",
        metavar="\b",
        default="-1",
        type=int,
        help="Deprecated by --instance-idx.",
    )


def _process_deprecated_params(args, logger):
    if args.node_id != -1:
        logger.warning("Argument --node_id is deprecated by --nodes-list.")
        args.nodes_list = str(args.node_id)
    if args.core_list != "":
        logger.warning("Argument --core_list is deprecated by --cores-list.")
        args.cores_list = args.core_list
    if args.use_logical_core:
        logger.warning(
            "Argument --use_logical_core is deprecated by --use-logical-cores."
        )
        args.use_logical_cores = args.use_logical_core
    if args.log_path != "":
        logger.warning("Argument --log_path is deprecated by --log-dir.")
        args.log_dir = args.log_path

    if args.multi_instance:
        logger.warning(
            "Argument --multi_instance is deprecated. Will be removed."  # noqa: G003
            + "If you are using the deprecated argument, please update it to the new one."
        )

    if args.enable_tcmalloc or args.enable_jemalloc or args.use_default_allocator:
        logger.warning(
            "Arguments --enable_tcmalloc, --enable_jemalloc and --use_default_allocator"  # noqa: G003
            + "are deprecated by --memory-allocator tcmalloc/jemalloc/auto."
        )
        if args.use_default_allocator:
            args.memory_allocator = "default"
        if args.enable_jemalloc:
            args.memory_allocator = "jemalloc"
        if args.enable_tcmalloc:
            args.memory_allocator = "tcmalloc"
    if args.disable_numactl:
        logger.warning(
            "Argument --disable_numactl is deprecated by --multi-task-manager taskset."
        )
        args.multi_task_manager = "taskset"
    if args.disable_taskset:
        logger.warning(
            "Argument --disable_taskset is deprecated by --multi-task-manager numactl."
        )
        args.multi_task_manager = "numactl"
    if args.disable_iomp:
        logger.warning(
            "Argument --disable_iomp is deprecated by --omp-runtime default."
        )
        args.omp_runtime = "default"
    if args.skip_cross_node_cores:
        logger.warning(
            "Argument --skip-cross-node-cores is deprecated by --bind-numa-node."
        )
        args.bind_numa_node = args.skip_cross_node_cores
    if args.rank != -1:
        logger.warning("Argument --rank is deprecated by --instance-idx.")
        args.instance_idx = str(args.rank)


def _init_parser(parser):
    """
    Helper function parsing the command line options
    @retval ArgumentParser
    """

    parser.add_argument(
        "-m",
        "--module",
        default=False,
        action="store_true",
        help="Changes each process to interpret the launch script "
        'as a python module, executing with the same behavior as "python -m".',
    )
    parser.add_argument(
        "--no-python",
        "--no_python",
        default=False,
        action="store_true",
        help="Avoid applying python to execute program.",
    )
    parser.add_argument(
        "--log-dir",
        "--log_dir",
        default="",
        type=str,
        help="The log file directory. Setting it to empty disables logging to files.",
    )
    parser.add_argument(
        "--log-file-prefix",
        "--log_file_prefix",
        default="run",
        type=str,
        help="log file name prefix",
    )
    parser.add_argument(
        "program",
        type=str,
        help="Full path to the program/script to be launched. "
        "followed by all the arguments for the script",
    )
    parser.add_argument(
        "program_args",
        nargs=argparse.REMAINDER,
    )
    launcher_multi_instances = MultiInstancesLauncher()
    launcher_multi_instances.add_common_params(parser)
    launcher_multi_instances.add_params(parser)
    _add_deprecated_params(parser)
    return parser


def _run_main_with_args(args, logger, logger_format_str):
    if platform.system() == "Windows":
        raise RuntimeError("Windows platform is not supported!!!")

    launcher_multi_instances = MultiInstancesLauncher(logger)

    _process_deprecated_params(args, logger)
    if args.log_dir:
        path = os.path.dirname(
            args.log_dir if args.log_dir.endswith("/") else f"{args.log_dir}/"
        )
        if not os.path.exists(path):
            os.makedirs(path)
        args.log_dir = path

        args.log_file_prefix = (
            f'{args.log_file_prefix}_{datetime.now().strftime("%Y%m%d%H%M%S")}'
        )
        fileHandler = logging.FileHandler(
            f"{args.log_dir}/{args.log_file_prefix}_instances.log"
        )
        logFormatter = logging.Formatter(logger_format_str)
        fileHandler.setFormatter(logFormatter)
        logger.addHandler(fileHandler)
    else:
        args.log_path = os.devnull

    assert args.no_python or args.program.endswith(
        ".py"
    ), 'For non Python script, you should use "--no-python" parameter.'

    env_before = set(os.environ.keys())
    if "LD_PRELOAD" in os.environ:
        lst_valid = []
        tmp_ldpreload = os.environ["LD_PRELOAD"]
        for item in tmp_ldpreload.split(":"):
            if item != "":
                matches = glob.glob(item)
                if len(matches) > 0:
                    lst_valid.append(item)
                else:
                    logger.warning(
                        f"{item} doesn't exist. Removing it from LD_PRELOAD."  # noqa: G004
                    )
        if len(lst_valid) > 0:
            os.environ["LD_PRELOAD"] = ":".join(lst_valid)
        else:
            os.environ["LD_PRELOAD"] = ""

    launcher = launcher_multi_instances
    launcher.launch(args)
    for x in sorted(set(os.environ.keys()) - env_before):
        logger.debug(f"{x}={os.environ[x]}")  # noqa: G004


def main():
    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=format_str)
    logger = logging.getLogger("torch-xeon-launcher")

    parser = argparse.ArgumentParser(
        description="This is a script for launching PyTorch inference on Intel® Xeon® Scalable Processors "
        + "with optimal configurations. Single instance inference, multi-instance inference are supported.\n",
        formatter_class=RawTextHelpFormatter,
    )
    parser = _init_parser(parser)
    args = parser.parse_args()
    _run_main_with_args(args, logger, format_str)


if __name__ == "__main__":
    main()
