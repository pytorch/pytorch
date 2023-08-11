"""
This is a script for launching PyTorch inference on Intel(R) Xeon(R) Scalable Processors with optimal configurations.
Single instance inference, multi-instance inference are enabled.

Note: term "instance" here doesn't refer to a cloud instance. This script is executed as a single process. It invokes
multiple "instances" which are formed from multiple threads for each. "instance" is kind of group of threads in this
context.

Illustrated as below:

::

    +-----------------------------+----------------------+-------+
    |            process          |        thread        | core  |
    +=============================+======================+=======+
    | torch.backends.xeon.run_cpu | instance 0: thread 0 |   0   |
    |                             |             thread 1 |   1   |
    |                             +----------------------+-------+
    |                             | instance 1: thread 0 |   2   |
    |                             |             thread 1 |   3   |
    |                             +----------------------+-------+
    |                             | ...                  |  ...  |
    |                             +----------------------+-------+
    |                             | instance N: thread 0 |   M   |
    |                             |             thread 1 |  M+1  |
    +-----------------------------+----------------------+-------+

To get the peak performance on Intel(R) Xeon(R) Scalable Processors, the script optimizes the configuration of thread and memory
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

How to use this module:
~~~~~~~~~~~~~~~~~~~~~~~

Single instance inference
-------------------------

1. Run single-instance inference on a single node with all CPU nodes.

::

   python -m torch.backends.xeon.run_cpu --throughput-mode script.py args

2. Run single-instance inference on a single CPU node.

::

   python -m torch.backends.xeon.run_cpu --node-id 1 script.py args

Multi-instance inference
------------------------

1. Multi-instance
   By default this tool runs one process per node. If you want to set the instance numbers and core per instance,
   --ninstances and  --ncores-per-instance should be set.

::

   python -m torch.backends.xeon.run_cpu -- python_script args

   eg: on an Intel(R) Xeon(R) Scalable Processor with 14 instance, 4 cores per instance

::

   python -m torch.backends.xeon.run_cpu --ninstances 14 --ncores-per-instance 4 python_script args

2. Run single-instance inference among multiple instances.
   By default, runs all ninstances. If you want to independently run a single instance among ninstances, specify rank.

   eg: run 0th instance on an Intel(R) Xeon(R) Scalable Processor with 2 instance (i.e., numactl -C 0-27)

::

   python -m torch.backends.xeon.run_cpu --ninstances 2 --rank 0 python_script args

   eg: run 1st instance on an Intel(R) Xeon(R) Scalable Processor with 2 instance (i.e., numactl -C 28-55)

::

   python -m torch.backends.xeon.run_cpu --ninstances 2 --rank 1 python_script args

   eg: run 0th instance on an Intel(R) Xeon(R) Scalable Processor with 2 instance, 2 cores per instance,
   first four cores (i.e., numactl -C 0-1)

::

   python -m torch.backends.xeon.run_cpu --core-list "0, 1, 2, 3" --ninstances 2 --ncores-per-instance 2
   --rank 0 python_script args

3. To look up what optional arguments this module offers:

::

    python -m torch.backends.xeon.run_cpu --help

Memory allocator
----------------

"--enable-tcmalloc" and "--enable-jemalloc" can be used to enable different memory allcator.

"""

import glob
import logging
import os
import platform
import re
import subprocess
import sys
from argparse import ArgumentParser, RawTextHelpFormatter, REMAINDER
from os.path import expanduser
from typing import Dict, List

from torch.distributed.elastic.multiprocessing import start_processes, Std

format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=format_str)
logger = logging.getLogger(__name__)


class _CPUinfo:
    """
    Get CPU information, such as cores list and NUMA information.
    """

    def __init__(self, test_input=""):
        self.cpuinfo = []
        if platform.system() in ["Windows", "Darwin"]:
            raise RuntimeError(f"{platform.system()} is not supported!!!")
        elif platform.system() == "Linux":
            # Sample output of: `lscpu --parse=CPU,Core,Socket,Node`
            #
            # # The following is the parsable format, which can be fed to other
            # # programs. Each different item in every column has an unique ID
            # # starting from zero.
            # # CPU,Core,Socket,Node
            # 0,0,0,0
            # 1,1,0,0
            # ...
            if test_input == "":
                lscpu_cmd = ["lscpu", "--parse=CPU,Core,Socket,Node"]
                lscpu_info = subprocess.check_output(
                    lscpu_cmd, universal_newlines=True
                ).split("\n")
            else:
                lscpu_info = test_input.split("\n")

            # Get information about  cpu, core, socket and node
            for line in lscpu_info:
                pattern = r"^([\d]+,[\d]+,[\d]+,[\d]?)"
                regex_out = re.search(pattern, line)
                if regex_out:
                    self.cpuinfo.append(regex_out.group(1).strip().split(","))

            # physical cores := core column in lscpu output
            #  logical cores :=  cPU column in lscpu output
            self.node_nums = int(max([line[3] for line in self.cpuinfo])) + 1
            self.node_physical_cores: List[List[int]] = []  # node_id is index
            self.node_logical_cores: List[List[int]] = []  # node_id is index
            self.physical_core_node_map = {}  # physical core to numa node id
            self.logical_core_node_map = {}  # logical core to numa node id

            for node_id in range(self.node_nums):
                cur_node_physical_core = []
                cur_node_logical_core = []
                for cpuinfo in self.cpuinfo:
                    nid = cpuinfo[3] if cpuinfo[3] != "" else "0"
                    if node_id == int(nid):
                        if int(cpuinfo[1]) not in cur_node_physical_core:
                            cur_node_physical_core.append(int(cpuinfo[1]))
                            self.physical_core_node_map[int(cpuinfo[1])] = int(node_id)
                        cur_node_logical_core.append(int(cpuinfo[0]))
                        self.logical_core_node_map[int(cpuinfo[0])] = int(node_id)
                self.node_physical_cores.append(cur_node_physical_core)
                self.node_logical_cores.append(cur_node_logical_core)

    def _physical_core_nums(self):
        return len(self.node_physical_cores) * len(self.node_physical_cores[0])

    def _logical_core_nums(self):
        return len(self.node_logical_cores) * len(self.node_logical_cores[0])

    def get_node_physical_cores(self, node_id):
        if node_id < 0 or node_id > self.node_nums - 1:
            raise ValueError(
                f"Invalid node id: {node_id}. Valid node ids: {list(range(len(self.node_physical_cores)))}"
            )
        return self.node_physical_cores[node_id]

    def get_node_logical_cores(self, node_id):
        if node_id < 0 or node_id > self.node_nums - 1:
            raise ValueError(
                f"Invalid node id: {node_id}. Valid node ids: {list(range(len(self.node_physical_cores)))}"
            )
        return self.node_logical_cores[node_id]

    def get_all_physical_cores(self):
        all_cores = []
        for cores in self.node_physical_cores:
            all_cores.extend(cores)
        return all_cores

    def get_all_logical_cores(self):
        all_cores = []
        for cores in self.node_logical_cores:
            all_cores.extend(cores)
        return all_cores

    def numa_aware_check(self, core_list):
        """
        Check whether all cores in core_list are in the same NUMA node. cross NUMA will reduce performance.
        We strongly advice to not use cores on different nodes.
        """
        cores_numa_map = self.logical_core_node_map
        numa_ids = []
        for core in core_list:
            numa_id = cores_numa_map[core]
            if numa_id not in numa_ids:
                numa_ids.append(numa_id)
        if len(numa_ids) > 1:
            logger.warning(
                "Numa Aware: cores:%s on different NUMA nodes:%s. To avoid \
this behavior, please use --ncores-per-instance knob to make sure number of cores is divisible by --ncores-per-\
instance. Alternatively, please use --skip-cross-node-cores knob.",
                str(core_list),
                str(numa_ids),
            )
        if len(numa_ids) == 0:
            raise RuntimeError(
                "invalid number of NUMA nodes; please make sure numa_ids >= 1"
            )
        return numa_ids


class _Launcher:
    r"""
    Class for launcher
    """

    msg_lib_notfound = f"Unable to find the {{0}} library file lib{{1}}.so in $CONDA_PREFIX/lib or $VIRTUAL_ENV/lib \
or /.local/lib/ or /usr/local/lib/ or /usr/local/lib64/ or /usr/lib or /usr/lib64 or \
{expanduser('~')}/.local/lib/ so the LD_PRELOAD environment variable will not be set."

    def __init__(self):
        self.cpuinfo = _CPUinfo()

    def add_lib_preload(self, lib_type):
        """
        Enable TCMalloc/JeMalloc/intel OpenMP
        """
        library_paths = []
        if "CONDA_PREFIX" in os.environ:
            library_paths.append(f"{os.environ['CONDA_PREFIX']}/lib")
        if "VIRTUAL_ENV" in os.environ:
            library_paths.append(f"{os.environ['VIRTUAL_ENV']}/lib")

        library_paths += [
            f"{expanduser('~')}/.local/lib",
            "/usr/local/lib",
            "/usr/local/lib64",
            "/usr/lib",
            "/usr/lib64",
        ]

        lib_find = False
        lib_set = False
        for item in os.getenv("LD_PRELOAD", "").split(":"):
            if item.endswith(f"lib{lib_type}.so"):
                lib_set = True
                break
        if not lib_set:
            for lib_path in library_paths:
                library_file = os.path.join(lib_path, f"lib{lib_type}.so")
                matches = glob.glob(library_file)
                if len(matches) > 0:
                    ld_preloads = [f"{matches[0]}", os.getenv("LD_PRELOAD", "")]
                    os.environ["LD_PRELOAD"] = os.pathsep.join(
                        [p.strip(os.pathsep) for p in ld_preloads if p]
                    )
                    lib_find = True
                    break
        return lib_set or lib_find

    def is_numactl_available(self):
        numactl_available = False
        try:
            cmd = ["numactl", "-C", "0", "-m", "0", "hostname"]
            r = subprocess.run(
                cmd,
                env=os.environ,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if r.returncode == 0:
                numactl_available = True
        except Exception:
            pass
        return numactl_available

    def set_memory_allocator(
        self, enable_tcmalloc=True, enable_jemalloc=False, use_default_allocator=False
    ):
        """
        Enable TCMalloc/JeMalloc with LD_PRELOAD and set configuration for JeMalloc.
        By default, PTMalloc will be used for PyTorch, but TCMalloc and JeMalloc can get better
        memory reuse and reduce page fault to improve performance.
        """
        if enable_tcmalloc and enable_jemalloc:
            raise RuntimeError(
                "Unable to enable TCMalloc and JEMalloc at the same time."
            )

        if enable_tcmalloc:
            find_tc = self.add_lib_preload(lib_type="tcmalloc")
            if not find_tc:
                msg = f'{self.msg_lib_notfound} you can use "conda install -c conda-forge gperftools" to install {{0}}'
                logger.warning(msg.format("TCmalloc", "tcmalloc"))  # noqa: G001
            else:
                logger.info("Use TCMalloc memory allocator")

        elif enable_jemalloc:
            find_je = self.add_lib_preload(lib_type="jemalloc")
            if not find_je:
                msg = f'{self.msg_lib_notfound} you can use "conda install -c conda-forge jemalloc" to install {{0}}'
                logger.warning(msg.format("Jemalloc", "jemalloc"))  # noqa: G001
            else:
                logger.info("Use JeMalloc memory allocator")
                self.set_env(
                    "MALLOC_CONF",
                    "oversize_threshold:1,background_thread:true,metadata_thp:auto",
                )

        elif use_default_allocator:
            pass

        else:
            find_tc = self.add_lib_preload(lib_type="tcmalloc")
            if find_tc:
                logger.info("Use TCMalloc memory allocator")
                return
            find_je = self.add_lib_preload(lib_type="jemalloc")
            if find_je:
                logger.info("Use JeMalloc memory allocator")
                return
            logger.warning(
                """Neither TCMalloc nor JeMalloc is found in $CONDA_PREFIX/lib or $VIRTUAL_ENV/lib
                            or /.local/lib/ or /usr/local/lib/ or /usr/local/lib64/ or /usr/lib or /usr/lib64 or
                           %s/.local/lib/ so the LD_PRELOAD environment variable will not be set.
                           This may drop the performance""",
                expanduser("~"),
            )

    def log_env_var(self, env_var_name=""):
        if env_var_name in os.environ:
            logger.info("%s=%s", env_var_name, os.environ[env_var_name])

    def set_env(self, env_name, env_value):
        if not env_value:
            logger.warning("%s is None", env_name)
        if env_name not in os.environ:
            os.environ[env_name] = env_value
        elif os.environ[env_name] != env_value:
            logger.warning(
                "Overriding value with the one set in environment variable: %s. \
Value applied: %s. Value ignored: %s",
                env_name,
                os.environ[env_name],
                env_value,
            )
        self.log_env_var(env_name)

    # set_kmp_affinity is used to control whether to set KMP_AFFINITY or not.
    # In scenario that use all cores on all nodes, including logical cores, setting KMP_AFFINITY disables logical cores.
    # In this case, KMP_AFFINITY should not be set.
    def set_multi_thread_and_allocator(
        self,
        ncores_per_instance,
        disable_iomp=False,
        set_kmp_affinity=True,
        enable_tcmalloc=True,
        enable_jemalloc=False,
        use_default_allocator=False,
    ):
        """
        Set multi-thread configuration and enable Intel openMP and TCMalloc/JeMalloc.
        By default, GNU openMP and PTMalloc are used in PyTorch. but Intel openMP and TCMalloc/JeMalloc are better alternatives
        to get performance benefit.
        """
        self.set_memory_allocator(
            enable_tcmalloc, enable_jemalloc, use_default_allocator
        )
        self.set_env("OMP_NUM_THREADS", str(ncores_per_instance))
        if not disable_iomp:
            find_iomp = self.add_lib_preload(lib_type="iomp5")
            if not find_iomp:
                msg = f'{self.msg_lib_notfound} you can use "conda install mkl" to install {{0}}'
                logger.warning(msg.format("iomp", "iomp5"))  # noqa: G001
            else:
                logger.info("Using Intel OpenMP")
                if set_kmp_affinity:
                    self.set_env("KMP_AFFINITY", "granularity=fine,compact,1,0")
                self.set_env("KMP_BLOCKTIME", "1")
        self.log_env_var("LD_PRELOAD")

    r"""
     Launcher for single instance and multi-instance
     """

    def launch(self, args):
        cores = []
        set_kmp_affinity = True
        enable_taskset = False
        if args.core_list:  # user specify what cores will be used by params
            cores = [int(x) for x in args.core_list.split(",")]
            if args.ncores_per_instance == -1:
                raise RuntimeError(
                    'please specify the "--ncores-per-instance" if you have pass the --core-list params'
                )
            elif (
                args.ninstances > 1
                and args.ncores_per_instance * args.ninstances < len(cores)
            ):
                logger.warning(
                    "only first %s cores will be used, \
but you specify %s cores in core_list",
                    args.ncores_per_instance * args.ninstances,
                    len(cores),
                )
            else:
                args.ninstances = len(cores) // args.ncores_per_instance

        else:
            if args.use_logical_core:
                if args.node_id != -1:
                    cores = self.cpuinfo.get_node_logical_cores(args.node_id)
                else:
                    cores = self.cpuinfo.get_all_logical_cores()
                    # When using all cores on all nodes, including logical cores,
                    # setting KMP_AFFINITY disables logical cores. Thus, KMP_AFFINITY should not be set.
                    set_kmp_affinity = False
            else:
                if args.node_id != -1:
                    cores = self.cpuinfo.get_node_physical_cores(args.node_id)
                else:
                    cores = self.cpuinfo.get_all_physical_cores()
            if (
                not args.multi_instance
                and args.ninstances == -1
                and args.ncores_per_instance == -1
            ):
                args.ninstances = 1
                args.ncores_per_instance = len(cores)
            elif (
                args.multi_instance
                and args.ninstances == -1
                and args.ncores_per_instance == -1
            ):
                args.throughput_mode = True
            elif args.ncores_per_instance == -1 and args.ninstances != -1:
                if args.ninstances > len(cores):
                    raise RuntimeError(
                        f"there are {len(cores)} total cores but you specify {args.ninstances} ninstances; \
please make sure ninstances <= total_cores)"
                    )
                else:
                    args.ncores_per_instance = len(cores) // args.ninstances
            elif args.ncores_per_instance != -1 and args.ninstances == -1:
                if not args.skip_cross_node_cores:
                    args.ninstances = len(cores) // args.ncores_per_instance
                else:
                    ncore_per_node = len(self.cpuinfo.node_physical_cores[0])
                    num_leftover_cores = ncore_per_node % args.ncores_per_instance
                    if args.ncores_per_instance > ncore_per_node:
                        # too many ncores_per_instance to skip cross-node cores
                        logger.warning(
                            "there are %s core(s) per socket, but you specify %s ncores_per_instance and \
skip_cross_node_cores. Please make sure --ncores-per-instance < core(s) per \
socket",
                            ncore_per_node,
                            args.ncores_per_instance,
                        )
                        exit(-1)
                    elif num_leftover_cores == 0:
                        # aren't any cross-node cores
                        logger.info(
                            "--skip-cross-node-cores is set, but there are no cross-node cores."
                        )
                        args.ninstances = len(cores) // args.ncores_per_instance
                    else:
                        # skip cross-node cores
                        if args.ninstances != -1:
                            logger.warning(
                                "--skip-cross-node-cores is exclusive to --ninstances. --ninstances \
won't take effect even if it is set explicitly."
                            )

                        i = 1
                        leftover_cores = set()
                        while ncore_per_node * i <= len(cores):
                            leftover_cores.update(
                                cores[
                                    ncore_per_node * i
                                    - num_leftover_cores : ncore_per_node * i
                                ]
                            )
                            i += 1
                        cores = list(set(cores) - leftover_cores)
                        assert len(cores) % args.ncores_per_instance == 0
                        args.ninstances = len(cores) // args.ncores_per_instance
            else:
                if args.ninstances * args.ncores_per_instance > len(cores):
                    raise RuntimeError(
                        "Please make sure ninstances * ncores_per_instance <= total_cores"
                    )
            if args.latency_mode:
                logger.warning(
                    "--latency-mode is exclusive to --ninstances, --ncores-per-instance, --node-id and \
--use-logical-core. They won't take effect even they are set explicitly."
                )
                args.ncores_per_instance = 4
                cores = self.cpuinfo.get_all_physical_cores()
                args.ninstances = len(cores) // args.ncores_per_instance

            if args.throughput_mode:
                logger.warning(
                    "--throughput-mode is exclusive to --ninstances, --ncores-per-instance, --node-id and \
--use-logical-core. They won't take effect even they are set explicitly."
                )
                args.ninstances = self.cpuinfo.node_nums
                cores = self.cpuinfo.get_all_physical_cores()
                args.ncores_per_instance = len(cores) // args.ninstances

        if args.ninstances > 1 and args.rank != -1:
            logger.info(
                "assigning %s cores for instance %s",
                args.ncores_per_instance,
                args.rank,
            )

        if not args.disable_numactl:
            numactl_available = self.is_numactl_available()
            if not numactl_available:
                if not args.disable_taskset:
                    logger.warning(
                        "Core binding with numactl is not available. Disabling numactl and using taskset instead. \
                    This may affect performance in multi-socket system; please use numactl if memory binding is needed."
                    )
                    args.disable_numactl = True
                    enable_taskset = True
                else:
                    logger.warning(
                        "Core binding with numactl is not available, and --disable_taskset is set. \
                    Please unset --disable_taskset to use taskset instead of numactl."
                    )
                    exit(-1)

        if not args.disable_taskset:
            enable_taskset = True

        self.set_multi_thread_and_allocator(
            args.ncores_per_instance,
            args.disable_iomp,
            set_kmp_affinity,
            args.enable_tcmalloc,
            args.enable_jemalloc,
            args.use_default_allocator,
        )
        entrypoint = ""
        launch_args = {}
        launch_envs: Dict[int, Dict] = {}
        launch_tee = {}
        for i in range(args.ninstances):
            cmd = []
            cur_process_cores = ""
            if not args.disable_numactl or enable_taskset:
                if not args.disable_numactl:
                    cmd = ["numactl"]
                elif enable_taskset:
                    cmd = ["taskset"]
                cores = sorted(cores)
                if (
                    args.rank == -1
                ):  # sequentially assign ncores_per_instance to ninstances
                    core_list = cores[
                        i
                        * args.ncores_per_instance : (i + 1)
                        * args.ncores_per_instance
                    ]
                else:  # assign ncores_per_instance from rank
                    core_list = cores[
                        args.rank
                        * args.ncores_per_instance : (args.rank + 1)
                        * args.ncores_per_instance
                    ]

                core_ranges: List[Dict] = []
                for core in core_list:
                    if len(core_ranges) == 0:
                        range_elem = {"start": core, "end": core}
                        core_ranges.append(range_elem)
                    else:
                        if core - core_ranges[-1]["end"] == 1:
                            core_ranges[-1]["end"] = core
                        else:
                            range_elem = {"start": core, "end": core}
                            core_ranges.append(range_elem)
                for r in core_ranges:
                    cur_process_cores = f"{cur_process_cores}{r['start']}-{r['end']},"
                cur_process_cores = cur_process_cores[:-1]
                if not args.disable_numactl:
                    numa_params = f"-C {cur_process_cores} "
                    numa_ids = ",".join(
                        [
                            str(numa_id)
                            for numa_id in self.cpuinfo.numa_aware_check(core_list)
                        ]
                    )
                    numa_params += f"-m {numa_ids}"
                    cmd.extend(numa_params.split())
                elif enable_taskset:
                    taskset_params = f"-c {cur_process_cores} "
                    cmd.extend(taskset_params.split())
            with_python = not args.no_python
            if with_python:
                cmd.append(sys.executable)
                cmd.append("-u")
            if args.module:
                cmd.append("-m")
            cmd.append(args.program)
            cmd.extend(args.program_args)
            cmd_s = " ".join(cmd)
            logger.info(cmd_s)
            if entrypoint == "":
                entrypoint = cmd[0]
            del cmd[0]
            launch_args[i] = tuple(cmd)
            launch_envs[i] = {}
            launch_tee[i] = Std.ALL

            if args.rank != -1:  # launches single instance, rank, only
                break

        ctx = start_processes(
            name=args.log_file_prefix,
            entrypoint=entrypoint,
            args=launch_args,
            envs=launch_envs,
            log_dir=args.log_path,
            tee=launch_tee,
        )
        ctx.wait()


def _add_memory_allocator_params(parser):
    group = parser.add_argument_group("Memory Allocator Parameters")
    # allocator control
    group.add_argument(
        "--enable-tcmalloc",
        "--enable_tcmalloc",
        action="store_true",
        default=False,
        help="Enable tcmalloc allocator",
    )
    group.add_argument(
        "--enable-jemalloc",
        "--enable_jemalloc",
        action="store_true",
        default=False,
        help="Enable jemalloc allocator",
    )
    group.add_argument(
        "--use-default-allocator",
        "--use_default_allocator",
        action="store_true",
        default=False,
        help="Use default memory allocator",
    )


def _add_multi_instance_params(parser):
    group = parser.add_argument_group("Multi-instance Parameters")
    # multi-instance control
    group.add_argument(
        "--ncores-per-instance",
        "--ncores_per_instance",
        metavar="\b",
        default=-1,
        type=int,
        help="Cores per instance",
    )
    group.add_argument(
        "--ninstances",
        metavar="\b",
        default=-1,
        type=int,
        help="For multi-instance, you should give the cores number you used for per instance.",
    )
    group.add_argument(
        "--skip-cross-node-cores",
        "--skip_cross_node_cores",
        action="store_true",
        default=False,
        help="If specified --ncores-per-instance, skips cross-node cores.",
    )
    group.add_argument(
        "--rank",
        metavar="\b",
        default="-1",
        type=int,
        help="Specify instance index to assign ncores_per_instance for rank; \
otherwise ncores_per_instance will be assigned sequentially to ninstances. Please refer to \
https://github.com/intel/intel-extension-for-pytorch/blob/master/docs/tutorials/performance_tuning/launch_script.md",
    )
    group.add_argument(
        "--latency-mode",
        "--latency_mode",
        action="store_true",
        default=False,
        help="By default 4 core per instance and use all physical cores",
    )
    group.add_argument(
        "--throughput-mode",
        "--throughput_mode",
        action="store_true",
        default=False,
        help="By default one instance per node and use all physical cores",
    )
    group.add_argument(
        "--node-id",
        "--node_id",
        metavar="\b",
        default=-1,
        type=int,
        help="node id for multi-instance, by default all nodes will be used",
    )
    group.add_argument(
        "--use-logical-core",
        "--use_logical_core",
        action="store_true",
        default=False,
        help="Whether only use physical cores",
    )
    group.add_argument(
        "--disable-numactl",
        "--disable_numactl",
        action="store_true",
        default=False,
        help="Disable numactl",
    )
    group.add_argument(
        "--disable-taskset",
        "--disable_taskset",
        action="store_true",
        default=False,
        help="Disable taskset",
    )
    group.add_argument(
        "--core-list",
        "--core_list",
        metavar="\b",
        default=None,
        type=str,
        help='Specify the core list as "core_id, core_id, ....", otherwise, all the cores will be used.',
    )
    group.add_argument(
        "--log-path",
        "--log_path",
        metavar="\b",
        default="",
        type=str,
        help="The log file directory. Default path is "
        ", which means disable logging to files.",
    )
    group.add_argument(
        "--log-file-prefix",
        "--log_file_prefix",
        metavar="\b",
        default="run",
        type=str,
        help="log file prefix",
    )


def _add_kmp_iomp_params(parser):
    group = parser.add_argument_group("IOMP Parameters")
    group.add_argument(
        "--disable-iomp",
        "--disable_iomp",
        action="store_true",
        default=False,
        help="By default, we use Intel OpenMP and libiomp5.so will be add to LD_PRELOAD",
    )


def create_args(parser=None):
    """
    Helper function parsing the command line options
    @retval ArgumentParser
    """
    parser.add_argument(
        "--multi-instance",
        "--multi_instance",
        action="store_true",
        default=False,
        help="Enable multi-instance, by default one instance per node",
    )

    parser.add_argument(
        "-m",
        "--module",
        default=False,
        action="store_true",
        help="Changes each process to interpret the launch script "
        "as a python module, executing with the same behavior as"
        '"python -m".',
    )

    parser.add_argument(
        "--no-python",
        "--no_python",
        default=False,
        action="store_true",
        help='Do not prepend the --program script with "python" - just exec '
        "it directly. Useful when the script is not a Python script.",
    )

    _add_memory_allocator_params(parser)
    _add_kmp_iomp_params(parser)

    _add_multi_instance_params(parser)
    # positional
    parser.add_argument(
        "program",
        type=str,
        help="The full path to the program/script to be launched. "
        "followed by all the arguments for the script",
    )

    # rest from the training program
    parser.add_argument("program_args", nargs=REMAINDER)


def main(args):
    env_before = set(os.environ.keys())
    if platform.system() in ["Windows", "Darwin"]:
        raise RuntimeError(f"{platform.system()} is not supported!!!")

    if args.log_path:
        os.makedirs(args.log_path, exist_ok=True)
    else:
        args.log_path = os.devnull

    if args.latency_mode and args.throughput_mode:
        raise RuntimeError(
            "Either args.latency_mode or args.throughput_mode should be set"
        )

    if not args.no_python and not args.program.endswith(".py"):
        raise RuntimeError(
            'For non Python script, you should use "--no-python" parameter.'
        )

    # Verify LD_PRELOAD
    if "LD_PRELOAD" in os.environ:
        lst_valid = []
        tmp_ldpreload = os.environ["LD_PRELOAD"]
        for item in tmp_ldpreload.split(":"):
            matches = glob.glob(item)
            if len(matches) > 0:
                lst_valid.append(item)
            else:
                logger.warning("%s doesn't exist. Removing it from LD_PRELOAD.", item)
        if len(lst_valid) > 0:
            os.environ["LD_PRELOAD"] = ":".join(lst_valid)
        else:
            os.environ["LD_PRELOAD"] = ""

    launcher = _Launcher()
    launcher.launch(args)
    for x in sorted(set(os.environ.keys()) - env_before):
        logger.debug("%s=%s", x, os.environ[x])


if __name__ == "__main__":
    parser = ArgumentParser(
        description="This is a script for launching PyTorch inference on Intel(R) Xeon(R) Scalable "
        "Processors with optimal configurations. Single instance inference, "
        "multi-instance inference are enable. To get the peak performance on Intel(R) "
        "Xeon(R) Scalable Processors, the script optimizes the configuration "
        "of thread and memory management. For thread management, the script configures thread "
        "affinity and the preload of Intel OMP library. For memory management, it configures "
        "NUMA binding and preload optimized memory allocation library (e.g. tcmalloc, jemalloc) "
        "\n################################# Basic usage ############################# \n"
        "\n 1. single instance\n"
        "\n   >>> python -m torch.backends.xeon.run_cpu python_script args \n"
        "\n2. multi-instance \n"
        "\n   >>> python -m torch.backends.xeon.run_cpu --ninstances xxx "
        "--ncores-per-instance xx python_script args\n"
        "\n############################################################################# \n",
        formatter_class=RawTextHelpFormatter,
    )
    create_args(parser)
    args = parser.parse_args()
    main(args)
