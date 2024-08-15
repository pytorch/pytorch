# mypy: allow-untyped-defs
import itertools
import os
import platform
import re
import subprocess


# lscpu Examples
# # The following is the parsable format, which can be fed to other
# # programs. Each different item in every column has an unique ID
# # starting from zero.
# CPU NODE SOCKET CORE L1d:L1i:L2:L3 ONLINE    MAXMHZ   MINMHZ      MHZ
#   0    0      0    0 0:0:0:0          yes 5000.0000 800.0000 2400.000
#   1    0      0    0 0:0:0:0          yes 5000.0000 800.0000 2400.000
#   2    0      0    1 0:0:0:0          yes 5000.0000 800.0000 2400.000
#   3    0      0    1 0:0:0:0          yes 5000.0000 800.0000 2400.000
#   4    1      1    2 0:0:0:0          yes 5000.0000 800.0000 2400.000
#   5    1      1    2 0:0:0:0          yes 5000.0000 800.0000 2400.000
#   6    1      1    3 0:0:0:0          yes 5000.0000 800.0000 2400.000
#   7    1      1    3 0:0:0:0          yes 5000.0000 800.0000 2400.000

# CPU SOCKET CORE L1d:L1i:L2:L3 ONLINE    MAXMHZ   MINMHZ      MHZ
#   0      0    0 0:0:0:0          yes 5000.0000 800.0000 2400.000
#   1      0    0 0:0:0:0          yes 5000.0000 800.0000 2400.000
#   2      0    1 0:0:0:0          yes 5000.0000 800.0000 2400.000
#   3      0    1 0:0:0:0          yes 5000.0000 800.0000 2400.000
#   4      0    2 0:0:0:0          yes 5000.0000 800.0000 2400.000
#   5      0    2 0:0:0:0          yes 5000.0000 800.0000 2400.000
#   6      0    3 0:0:0:0          yes 5000.0000 800.0000 2400.000
#   7      0    3 0:0:0:0          yes 5000.0000 800.0000 2400.000

# CPU NODE SOCKET CORE L1d:L1i:L2:L3 ONLINE    MAXMHZ   MINMHZ      MHZ
#   0    0      0    0 0:0:0:0          yes 5000.0000 800.0000 2400.000
#   1    0      0    1 0:0:0:0          yes 5000.0000 800.0000 2400.000
#   2    1      1    2 0:0:0:0          yes 5000.0000 800.0000 2400.000
#   3    1      1    3 0:0:0:0          yes 5000.0000 800.0000 2400.000
#   4    0      0    0 0:0:0:0          yes 5000.0000 800.0000 2400.000
#   5    0      0    1 0:0:0:0          yes 5000.0000 800.0000 2400.000
#   6    1      1    2 0:0:0:0          yes 5000.0000 800.0000 2400.000
#   7    1      1    3 0:0:0:0          yes 5000.0000 800.0000 2400.000

# CPU NODE SOCKET CORE L1d:L1i:L2:L3 ONLINE    MAXMHZ   MINMHZ      MHZ
#   0    0      0    0 0:0:0:0          yes 5000.0000 800.0000 2400.000
#   1    0      0    0 0:0:0:0          yes 5000.0000 800.0000 2400.000
#   2    0      0    1 0:0:0:0          yes 5000.0000 800.0000 2400.000
#   3    0      0    1 0:0:0:0          yes 5000.0000 800.0000 2400.000
#   4    0      0    2 0:0:0:0          yes 5000.0000 800.0000 2400.000
#   5    0      0    2 0:0:0:0          yes 5000.0000 800.0000 2400.000
#   6    0      0    3 0:0:0:0          yes 5000.0000 800.0000 2400.000
#   7    0      0    3 0:0:0:0          yes 5000.0000 800.0000 2400.000
#   8    0      0    4 0:0:0:0          yes 3800.0000 800.0000 2400.000
#   9    0      0    5 0:0:0:0          yes 3800.0000 800.0000 2400.000
#  10    0      0    6 0:0:0:0          yes 3800.0000 800.0000 2400.000
#  11    0      0    7 0:0:0:0          yes 3800.0000 800.0000 2400.000


class CoreInfo:
    """
    Class to store core-specific information, including:
    - [int] CPU index
    - [int] Core index
    - [int] Numa node index
    - [int] Socket index
    - [bool] is a physical core or not
    - [float] maxmhz
    - [bool] is a performance core
    """

    def __init__(self, lscpu_txt="", headers=None) -> None:
        if headers is None:
            headers = {}
        self.cpu = -1
        self.core = -1
        self.socket = -1
        self.node = -1
        self.is_physical_core = True
        self.maxmhz = 0.0
        self.is_p_core = True
        if lscpu_txt != "" and len(headers) > 0:
            self.parse_raw(lscpu_txt, headers)

    def parse_raw(self, cols, headers):
        self.cpu = int(cols[headers["cpu"]])
        self.core = int(cols[headers["core"]])
        if "node" in headers:
            self.node = int(cols[headers["node"]])
            self.socket = int(cols[headers["socket"]])
        else:
            self.node = int(cols[headers["socket"]])
            self.socket = int(cols[headers["socket"]])
        if "maxmhz" in headers:
            self.maxmhz = float(cols[headers["maxmhz"]])

    def __str__(self):
        return f"{self.cpu}\t{self.core}\t{self.socket}\t{self.node}\t{self.is_physical_core}\t{self.maxmhz}\t{self.is_p_core}"


class CPUPool(list):
    """
    List of CoreInfo objects
    """

    def __init__(self) -> None:
        super().__init__(self)

    def get_ranges(self, l):
        for a, b in itertools.groupby(enumerate(l), lambda pair: pair[1] - pair[0]):
            bl = list(b)
            yield bl[0][1], bl[-1][1]

    def get_pool_txt(self, return_mode="auto"):
        cpu_ids = [c.cpu for c in self]
        cpu_ranges = list(self.get_ranges(cpu_ids))
        cpu_ids_txt = ",".join([str(c) for c in cpu_ids])
        cpu_ranges_txt = ",".join([f"{r[0]}-{r[1]}" for r in cpu_ranges])
        node_ids_txt = ",".join(
            [
                str(n)
                for n in sorted(list(set([c.node for c in self])))  # noqa: C414,C403
            ]
        )
        ret = {"cores": "", "nodes": node_ids_txt}
        if return_mode.lower() == "list":
            ret["cores"] = cpu_ids_txt
        elif return_mode.lower() == "range":
            ret["cores"] = cpu_ranges_txt
        else:
            if len(cpu_ids) <= len(cpu_ranges):
                ret["cores"] = cpu_ids_txt
            else:
                ret["cores"] = cpu_ranges_txt
        return ret


class CPUPoolList:
    """
    Get a CPU pool with all available CPUs and CPU pools filtered with designated criterias.
    """

    def __init__(self, logger=None, lscpu_txt="") -> None:
        self.pool_all = CPUPool()
        self.pools_ondemand: list[CPUPool] = []

        self.logger = logger
        if platform.system() == "Linux" and platform.machine() == "x86_64":
            # Retrieve CPU information from lscpu.
            if lscpu_txt.strip() == "":
                args = ["lscpu", "--all", "--extended"]
                my_env = os.environ.copy()
                my_env["LC_ALL"] = "C"
                lscpu_info = subprocess.check_output(
                    args, env=my_env, universal_newlines=True
                )
            else:
                lscpu_info = lscpu_txt

            # Filter out lines that are really useful.
            lst_lscpu_info = lscpu_info.strip().split("\n")
            headers = {}
            num_cols = 0
            for line in lst_lscpu_info:
                line = re.sub(" +", " ", line.lower().strip())
                if "cpu" in line and "socket" in line and "core" in line:
                    t = line.split(" ")
                    num_cols = len(t)
                    for i in range(num_cols):
                        if t[i] in ["cpu", "core", "socket", "node", "maxmhz"]:
                            headers[t[i]] = i
                else:
                    t = line.split(" ")
                    if (
                        len(t) == num_cols
                        and t[headers["cpu"]].isdigit()
                        and t[headers["core"]].isdigit()
                        and t[headers["socket"]].isdigit()
                    ):
                        self.pool_all.append(CoreInfo(t, headers))
            assert len(self.pool_all) > 0, "cpuinfo is empty"
        else:
            raise RuntimeError(f"Unsupported platform {platform.system()}")

        # Determine logical cores
        core_cur = -1
        self.pool_all.sort(key=lambda x: (x.core, x.cpu))
        for c in self.pool_all:
            if core_cur != c.core:
                core_cur = c.core
            else:
                c.is_physical_core = False
        self.pool_all.sort(key=lambda x: x.cpu)

        # Determine e cores
        maxmhzs = list(set([c.maxmhz for c in self.pool_all]))  # noqa: C403
        maxmhzs.sort()
        mmaxmhzs = max(maxmhzs)
        if mmaxmhzs > 0:
            maxmhzs_norm = [f / mmaxmhzs for f in maxmhzs]
            separator_idx = -1
            for i in range(1, len(maxmhzs_norm)):
                if maxmhzs_norm[i] - maxmhzs_norm[i - 1] >= 0.15:
                    separator_idx = i
                    break
            if separator_idx > -1:
                e_core_mhzs = maxmhzs[:separator_idx]
                for c in self.pool_all:
                    if c.maxmhz in e_core_mhzs:
                        c.is_p_core = False

    def verbose(self, level, msg, warning_type=None):
        if self.logger:
            logging_fn = {
                "warning": self.logger.warning,
                "info": self.logger.info,
            }
            assert (
                level in logging_fn.keys()
            ), f"Unrecognized logging level {level} is detected. Available levels are {logging_fn.keys()}."
            if warning_type:
                logging_fn[level](msg, _type=warning_type)
            else:
                logging_fn[level](msg)
        else:
            print(msg)

    """
    Generate CPU pools from all available CPU cores with designated criterias.
    - ninstances [int]:          Number of instances. Should be a non negative integer, 0 by default.
                                 When it is 0, it will be set according to usage scenarios automatically in the
                                 function.
    - ncores_per_instance [int]: Number of cores per instance. Should be a non negative integer, 0 by default.
                                 When it is 0, it will be set according to usage scenarios automatically in the
                                 function.
    - use_logical_cores [bool]:  Use logical cores on the workloads or not, False by default. When set to False,
                                 only physical cores are used.
    - use_e_cores [bool]:        Whether to use Efficient-Cores, False by default.
                                        Efficient-Cores Performance-Cores
                                  True:        X               X
                                 False:                        X
    - bind_numa_node [bool]:     Bind instances to be executed on cores on a single NUMA node, False by default.
    - strategy [str]:            Instructs how to bind numa nodes. Can be 'close' or 'scatter', 'close' by default.
                                      Node: | -------------- 0 --------------- | --------------- 1 -------------- |
                                      Core: 00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23
                                   'close': { --- ins 0 --- } { --- ins 1 --- }
                                 'scatter': { --- ins 0 --- }                   { --- ins 1 --- }
    - nodes_list [list]:         A list containing all node ids that the execution is expected to be running on.
    - cores_list [list]:         A list containing all cpu ids that the execution is expected to be running on.
    - return_mode [str]:         A string that defines how result values are formed, can be 'auto', 'list' or 'range'.
                                 'list':  a string with comma-separated cpu ids, '0,1,2,3,...', is returned.
                                 'range': a string with comma-separated cpu id ranges, '0-2,6-8,...', is returned.
                                 'auto':  a 'list' or a 'range' whichever has less number of elements that are separated
                                          by comma is returned. I.e. for a list '0,1,2,6,7,8' and a range '0-2,6-8',
                                          both reflect the same cpu configuration, the range '0-2,6-8' is returned.
    """

    def gen_pools_ondemand(
        self,
        ninstances=0,
        ncores_per_instance=0,
        use_logical_cores=False,
        use_e_cores=False,
        bind_numa_node=False,
        strategy="close",
        nodes_list=None,
        cores_list=None,
        return_mode="auto",
    ):
        if nodes_list is None:
            nodes_list = []
        if cores_list is None:
            cores_list = []

        # Generate an aggregated CPU pool
        if len(cores_list) > 0:
            cores_available = [c.cpu for c in self.pool_all]
            assert set(cores_list).issubset(
                set(cores_available)
            ), f"Designated cores list {cores_list} contains invalid cores."
            pool = [c for c in self.pool_all if c.cpu in cores_list]
        else:
            if len(nodes_list) > 0:
                nodes_available = set([c.node for c in self.pool_all])  # noqa: C403
                assert set(nodes_list).issubset(
                    nodes_available
                ), f"Designated nodes list {nodes_list} contains invalid nodes out from {nodes_available}."
                pool = [c for c in self.pool_all if c.node in nodes_list]
            else:
                pool = self.pool_all
        if not use_logical_cores:
            pool = [c for c in pool if c.is_physical_core]
            logical_cores = [c.cpu for c in pool if not c.is_physical_core]
            if len(logical_cores) > 0:
                self.verbose(
                    "info",
                    f"Logical cores are detected ({logical_cores}). Disabled for performance consideration. "
                    + "You can enable them with argument --use-logical-cores.",
                )
        if not use_e_cores:
            pool = [c for c in pool if c.is_p_core]
            e_cores = [c.cpu for c in pool if not c.is_p_core]
            if len(e_cores) > 0:
                self.verbose(
                    "info",
                    f"Efficient-Cores are detected ({e_cores}). Disabled for performance consideration. "
                    + "You can enable them with argument --use-e-cores.",
                )

        # Determine ninstances and ncores_per_instance for grouping
        assert (
            ncores_per_instance >= -1
        ), "Argument --ncores-per-instance cannot be a negative value other than -1."
        assert ninstances >= 0, "Argument --ninstances cannot be a negative value."
        pool.sort(key=lambda x: (x.core, 1 - int(x.is_physical_core)))
        nodes = list(set([c.node for c in pool]))  # noqa: C403
        is_greedy = False
        if ncores_per_instance == -1:
            is_greedy = True
            ncores_per_instance = 0
        if ncores_per_instance + ninstances == 0:
            # Both ncores_per_instance and ninstances are 0
            ninstances = 1

        rst = []
        if ncores_per_instance == 0:
            pool_process = []
            ninstances_node = []
            if bind_numa_node:
                for node in nodes:
                    pool_node = [c for c in pool if c.node == node]
                    pool_process.append(pool_node)
                    ninstances_local = (ninstances * len(pool_node)) // len(pool)
                    if (ninstances_local) == 0 or (
                        (ninstances * len(pool_node)) % len(pool) > 0
                    ):
                        ninstances_local += 1
                    ninstances_node.append(ninstances_local)
                for _ in range(int(sum(ninstances_node)) - ninstances):
                    ncores_per_instance_local = []
                    for i in range(len(nodes)):
                        ncores_node = len([c for c in pool if c.node == nodes[i]])
                        tmp = ncores_node / ninstances_node[i]
                        if ninstances_node[i] == 1:
                            tmp = len(pool)
                        ncores_per_instance_local.append(tmp)
                    ncores_per_instance_local_min = min(ncores_per_instance_local)
                    if ncores_per_instance_local_min == len(pool):
                        break
                    index = ncores_per_instance_local.index(
                        ncores_per_instance_local_min
                    )
                    ninstances_node[index] -= 1
                delta = int(sum(ninstances_node)) - ninstances
                if delta > 0:
                    ncores_per_instance_local = []
                    for i in range(len(nodes)):
                        ncores_per_instance_local.append(
                            {
                                "index": i,
                                "count": len([c for c in pool if c.node == nodes[i]]),
                            }
                        )
                    ncores_per_instance_local.sort(
                        key=lambda x: (x["count"], len(nodes) - x["index"])
                    )
                    for i in range(delta):
                        ninstances_node[ncores_per_instance_local[i]["index"]] -= 1
            else:
                pool_process.append(pool)
                ninstances_node.append(ninstances)
            for i in range(len(pool_process)):
                p = pool_process[i]
                n = ninstances_node[i]
                if n == 0:
                    continue
                tmp = []
                for j in range(n):
                    tmp.append({"ncores": len(p) // n, "pool": []})
                if is_greedy:
                    ncores_residual = len(p) % n
                    for j in range(ncores_residual):
                        tmp[j]["ncores"] += 1
                ncores_assigned = 0
                for j in range(len(tmp)):
                    tmp[j]["pool"] = p[
                        ncores_assigned : ncores_assigned + tmp[j]["ncores"]
                    ]
                    ncores_assigned += tmp[j]["ncores"]
                rst += tmp
        else:
            pool_process = []
            if bind_numa_node:
                for node in nodes:
                    pool_process.append([c for c in pool if c.node == node])
            else:
                pool_process.append(pool)
            for i in range(len(pool_process)):
                p = pool_process[i]
                n = len(p) // ncores_per_instance
                ncores_assigned = 0
                for _ in range(n):
                    item = {"ncores": 0, "node": nodes[i], "pool": []}
                    item["ncores"] = ncores_per_instance
                    item["pool"] = p[
                        ncores_assigned : ncores_assigned + ncores_per_instance
                    ]
                    ncores_assigned += ncores_per_instance
                    rst.append(item)
            if ninstances > 0:
                assert ninstances <= len(rst), (
                    f"Requested --ninstances ({ninstances}) and --ncores_per_instance ({ncores_per_instance}) "
                    + "combination is not supported. Please adjust either or both of these 2 parameters and try again."
                )
                if ninstances < len(rst):
                    if strategy == "close":
                        rst = rst[:ninstances]
                    elif strategy == "scatter":
                        if len(pool_process) == 1:
                            step = len(rst) // ninstances
                            if len(rst) % ninstances > 0:
                                step += 1
                            rst = rst[::step]
                        else:
                            rst_map = []
                            ninstances_node_avai = []
                            ninstances_node = []
                            for node in nodes:
                                tmp = [r for r in rst if r["node"] == node]
                                rst_map.append(tmp)
                                ninstances_node_avai.append(len(tmp))
                                ninstances_node.append(0)
                            index = 0
                            for _ in range(ninstances):
                                while index < len(nodes):
                                    index += 1
                                    if index == len(nodes):
                                        index = 0
                                    if ninstances_node_avai[index - 1] > 0:
                                        ninstances_node[index - 1] += 1
                                        ninstances_node_avai[index - 1] -= 1
                                        break
                            rst.clear()
                            for i in range(len(ninstances_node)):
                                rst += rst_map[i][: ninstances_node[i]]
                    else:
                        raise ValueError(f"Strategy {strategy} is not available.")

        # Split the aggregated pool into individual pools
        self.pools_ondemand.clear()
        for item in rst:
            # Generate individual raw pool
            pool_local = CPUPool()
            for c in item["pool"]:
                pool_local.append(c)
            pool_local.sort(key=lambda x: x.cpu)
            self.pools_ondemand.append(pool_local)
