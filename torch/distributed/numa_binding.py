"""
This script is used for binding the rank-process to cpu-cores.
The current binding options are as follows:
1. Node (node)
2. Socket (called socket)
3. Exclusive (called exclusive)
4. Core-Complex (Called core-complex)
"""
import subprocess
import os
import socket
from argparse import ArgumentParser, REMAINDER
from collections import defaultdict
import re
import shutil
from torch.distributed.elastic.utils.logging import get_logger
import pynvml

# Commands/Paths to get system-level information
NUMA_CMD = "/sys/bus/pci/devices/{value}/numa_node"
NUMA_CPU_MAP_CMD = "/sys/devices/system/node/node{value}/cpumap"
CPU_MAP_CMD = "/proc/self/status | grep \"Cpus_allowed:\""
THREAD_SIBLINGS_CMD = "/sys/devices/system/cpu/cpu{value}/topology/thread_siblings"
SHARED_CACHE_CMD = "/sys/devices/system/node/node{value_1}/cpu{value_2}/cache/{value_3}/shared_cpu_map"
CACHE_CMD = "/sys/devices/system/node/node{value_1}/cpu{value_2}/cache"
SOCKET_CMD = "/sys/devices/system/node/node{value}/cpulist"
PHYSICAL_PACKAGE_ID_CMD = "/sys/devices/system/cpu/cpu{value}/topology/physical_package_id"
POSSIBLE_NODES_CMD = "/sys/devices/system/node/possible"

logger = get_logger(__name__)


class System:
    """
    Abstracts system specific methods, so it could be mocked for unit tests, across various different types
    of systems.
    """

    def __init__(self):
        pass

    def execute_command(self, cmd):
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
        outs, errs = process.communicate()
        if process.returncode != 0:
            raise Exception(f"command:{cmd}, return code:{process.returncode},stderr:{errs}")
        return outs.decode('UTF-8')

    # returns number of gpus
    def get_gpu_count(self):
        # Initialize NVML
        pynvml.nvmlInit()
        # Get the number of GPU devices
        device_count = pynvml.nvmlDeviceGetCount()
        # Shutdown NVML
        pynvml.nvmlShutdown()
        # outs = self.execute_command(NUM_GPUS_CMD)
        return int(device_count)

    # returns array indexed by GPU id and mapping to value NUMA node id
    def get_numa_nodes(self):
        numaNodes = []
        # Initialize NVML
        pynvml.nvmlInit()
        # Get the number of GPU devices
        device_count = pynvml.nvmlDeviceGetCount()
        # Retrieve and print PCI bus ID for each GPU
        pciBusIDs = []
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            pci_info = pynvml.nvmlDeviceGetPciInfo(handle)
            bus_id = pci_info.busId.decode()  # Decode bytes to string
            pciBusIDs.append(bus_id)
        # Shutdown NVML
        pynvml.nvmlShutdown()
        for busID in pciBusIDs:
            pciFields = busID.split(":")
            pciDir = f"{pciFields[0][-4:]}:{pciFields[1]}:{pciFields[2]}"
            numaFile = NUMA_CMD.format(value=pciDir.lower())
            try:
                with open(numaFile) as numa_node_text:
                    node = int(numa_node_text.read())
                    numaNodes.append(node)
            except FileNotFoundError:
                print(f"The file {numaFile} does not exist.")
        return numaNodes

    # returns full set of CPUs affined to the NUMA node
    # includes reserved and non-reserved CPUs
    def get_numa_node_to_cpus(self, affinedNumaNode):
        numaCpuMapCmd = NUMA_CPU_MAP_CMD.format(value=affinedNumaNode)
        try:
            with open(numaCpuMapCmd) as file:
                outs = file.read()
        except FileNotFoundError:
            print(f"The file {numaCpuMapCmd} does not exist.")
        numaCpuMap = outs.split(",")
        numaCpuMapHex = "0x{}".format("".join(numaCpuMap))
        numaCpuMapVal = int(numaCpuMapHex, 16)
        return numaCpuMapVal

    # returns a bitmap for each core, its sibling cores
    def get_thread_siblings(self, cpu):
        threadsFile = THREAD_SIBLINGS_CMD.format(value=cpu)
        try:
            with open(threadsFile) as threads:
                threadsMap = threads.read()
                threadsMapHex = "0x{}".format("".join(threadsMap.split(",")))
                threadsMapVal = int(threadsMapHex, 16)
        except FileNotFoundError:
            print(f"The file {threadsFile} does not exist.")

        return threadsMapVal

    # get a bitmap of cpus that are allowed to the process
    def get_available_cpu(self):
        with open('/proc/self/status', 'r') as file:
            for line in file:
                if "Cpus_allowed:" in line:
                    outs = line.strip()
                    break
        cpuMap = outs.split()[1].split(",")
        cpuMapHex = "0x{}".format("".join(cpuMap))
        cpuMapVal = int(cpuMapHex, 16)
        return cpuMapVal

    # get a sorted list of shared caches for the given cpu in the given node
    def get_shared_caches(self, node, cpuid):
        rootdir = CACHE_CMD.format(value_1=node, value_2=cpuid)
        caches = []
        for file in os.listdir(rootdir):
            d = os.path.join(rootdir, file)
            if os.path.isdir(d) and len(file) >= 5 and file[:5] == 'index':
                with open(os.path.join(d, "type")) as cacheTypeInfo:
                    cacheType = cacheTypeInfo.read()
                    if cacheType.find("Unified") != -1 or cacheType.find("Data") != -1:
                        caches.append(file)
        return sorted(caches)

    # returns for a cpu on a node, a bitmap of cpus sharing its given shared cache
    def get_cpus_in_shared_cache(self, node, cpuid, cacheName):
        sharedCacheCmd = SHARED_CACHE_CMD.format(value_1=node, value_2=cpuid, value_3=cacheName)
        try:
            with open(sharedCacheCmd) as cache_info:
                cpuInSharedCacheMap = cache_info.read()
                cpuInSharedCacheMapHex = "0x{}".format("".join(cpuInSharedCacheMap.split(",")))
                cpuInSharedCacheVal = int(cpuInSharedCacheMapHex, 16)
        except FileNotFoundError:
            print(f"The file {sharedCacheCmd.format(node, cpuid, cacheName)} does not exist.")
        return cpuInSharedCacheVal

    # returns a map for each socket its nodes
    def get_socket_node(self):
        try:
            with open(POSSIBLE_NODES_CMD) as file:
                num_nodes = file.read()
        except FileNotFoundError:
            print(f"The file {POSSIBLE_NODES_CMD} does not exist.")

        num_nodes = re.split('-', num_nodes)
        socket_node = defaultdict(list)
        for n in range(0, int(num_nodes[1]) + 1):
            socketCmd = SOCKET_CMD.format(value=n)
            try:
                with open(socketCmd) as file:
                    print_node = file.read()
            except FileNotFoundError:
                print(f"The file {socketCmd} does not exist.")
            print_node = re.split('-', print_node)
            print_node[1] = print_node[1].split(',')
            try:
                physicalPackageIDCmd = PHYSICAL_PACKAGE_ID_CMD.format(value=print_node[0])
                with open(physicalPackageIDCmd) as file:
                    socket_id = file.read()
            except FileNotFoundError:
                print(f"The file {physicalPackageIDCmd} does not exist.")
            if socket_id not in socket_node.keys():
                socket_node[socket_id].append(n)
            else:
                socket_node[socket_id].append(n)
        return socket_node


def parse_args():
    """
    Helper function parsing the command line options
    @retval ArgumentParser
    """
    parser = ArgumentParser(description="This script is used for binding NUMA nodes with the program of choice. ")
    parser.add_argument("--affinity", type=str,
                        help="NUMA Config needed to launch (node,exclusive,core-complex,socket)")
    parser.add_argument("--local_rank", type=int, required=False, help="Rank on node")
    # positional
    parser.add_argument("training_executable", type=str,
                        help="This contains the executable needed to run the program (eg: python,gcc,g++,nvcc)")
    # Training Script added as an argument to the NUMA script
    parser.add_argument('training_script_args', nargs=REMAINDER,
                        help="This contains the rest of the launching script apart from the executable")
    return parser.parse_args()


def get_local_rank(args):
    local_rank = None
    is_rank_required = (args.local_rank is not None)
    local_rank = os.environ['LOCAL_RANK']
    return int(local_rank), is_rank_required


def check_numactl():
    # Check if numactl is installed
    numactl_path = shutil.which('numactl')
    if not numactl_path:
        # Throw an error if numactl is not available
        raise RuntimeError("numactl is not installed")
    return numactl_path


def get_cmd(args, numactlargs, local_rank, is_rank_required):
    numactl_path = check_numactl()
    cmd = [numactl_path] \
          + numactlargs \
          + [args.training_executable] + args.training_script_args
    if is_rank_required:
        cmd = cmd + ["--local_rank={}".format(local_rank)]
    return cmd


def run_cmd(cmd, env):
    processes = []
    process = subprocess.Popen(cmd, env=env)
    processes.append(process)
    for process in processes:
        process.wait()


class Numa:
    """
    Base class for numa binding methods. defines common functions
    """

    def __init__(self, local_rank, system):
        self.local_rank = local_rank
        self.system = system
        gpu_count = system.get_gpu_count()
        if local_rank > gpu_count:
            raise Exception("Local Rank is greater than the number of GPUs")

    def get_affined_gpus(self, numa_nodes):
        affinedNumaNode = numa_nodes[self.local_rank]
        affinedGpuIndex = 0
        affinedGpuCount = 0
        for i in range(len(numa_nodes)):
            if numa_nodes[i] == affinedNumaNode:
                if i == self.local_rank:
                    affinedGpuIndex = affinedGpuCount
                affinedGpuCount += 1
        return affinedGpuCount, affinedGpuIndex

    def get_args_string(self, core_ranges, key_index):
        ranges_join = ','.join(core_ranges)
        phys_cpu = "--physcpubind="
        numactlargs = []
        numactlargs_phys_cpu = phys_cpu + ranges_join
        numactlargs += [numactlargs_phys_cpu, "--membind={}".format(key_index)]
        return numactlargs

    def rangify(self, cpulist):
        ranges = []
        first = last = cpulist[0]
        for index in cpulist[1:]:
            if index - last > 1:
                if first == last:
                    ranges.append("{}".format(first))
                else:
                    ranges.append("{}-{}".format(first, last))
                first = index
            last = index
        if first == last:
            ranges.append("{}".format(first))
        else:
            ranges.append("{}-{}".format(first, last))
        return ranges

    def get_numactl_args(self):
        return []


class Node(Numa):
    """
    implements node numa-binding
    """

    def __init__(self, local_rank, system):
        super().__init__(local_rank, system)

    def get_numactl_args(self):
        numa_gpu = self.system.get_numa_nodes()
        return ["--cpunodebind={}".format(int(numa_gpu[self.local_rank])),
                "--membind={}".format(int(numa_gpu[self.local_rank]))]


class Exclusive(Numa):
    """
    implements exclusive numa-binding
    """

    def __init__(self, local_rank, system):
        super().__init__(local_rank, system)

    def get_numactl_args(self):
        # Step 1: get the number of gpus
        numGPUs = self.system.get_gpu_count()

        # Step 2: find numa nodes
        numaNodes = self.system.get_numa_nodes()
        # Step 3: find affinedNumaNode, number of gpus affined to that node
        # and the relative index of the gpu on that node
        affinedNumaNode = numaNodes[self.local_rank]
        affinedGpuCount, affinedGpuIndex = self.get_affined_gpus(numaNodes)

        # Step 4: Get all cpus affined to the numa node
        numaCpuMapVal = self.system.get_numa_node_to_cpus(affinedNumaNode)

        # Step 5: Get CPUs available
        cpuMapVal = self.system.get_available_cpu()

        # Step 6: find available cpus on the numa node
        numaCpus = numaCpuMapVal & cpuMapVal

        # Step 7: find number of cpus
        numaCpuLen = len(bin(numaCpus)) - 2

        # Step 8: get cpu ids on numa node
        cpuList = []
        for i in range(numaCpuLen):
            if (numaCpus >> i) & 1 == 1:
                cpuList.append(i)

        # Step 9: get list of cpus that are fully available
        # considering thread siblings
        coreIDs = {}
        cpuMask = 0
        for i in cpuList:
            if cpuMask & (1 << i) == 0:
                threadsMapVal = self.system.get_thread_siblings(i)
                coreID = (threadsMapVal & (-threadsMapVal)).bit_length() - 1
                if (numaCpus & threadsMapVal) == threadsMapVal:
                    coreIDs[coreID] = threadsMapVal
                cpuMask |= threadsMapVal

        # Step 10: allocate the cores to this gpu
        # based on relative index on this node
        numCpus = len(coreIDs) // affinedGpuCount
        startIndex = numCpus * affinedGpuIndex
        endIndex = startIndex + numCpus
        resultCpuVal = 0
        count = 0
        for i in sorted(coreIDs.keys()):
            if count >= startIndex and count < endIndex:
                resultCpuVal |= coreIDs[i]
            count += 1

        # Step 11: create a list of cpus
        resultCpuLen = len(bin(resultCpuVal)) - 2
        resultCpuList = []
        for i in range(resultCpuLen):
            if (resultCpuVal >> i) & 1 == 1:
                resultCpuList.append(i)

        # Step 12: range-ify the cpu list
        cpu_ranges = self.rangify(resultCpuList)
        numactlargs = self.get_args_string(cpu_ranges, affinedNumaNode)
        return numactlargs


class CoreComplex(Numa):
    """
    implements core-complex numa-binding
    """

    def __init__(self, local_rank, system):
        super().__init__(local_rank, system)

    def get_numactl_args(self):
        # Step 1: get the number of gpus
        numGPUs = self.system.get_gpu_count()

        # Step 2: find numa nodes
        numaNodes = self.system.get_numa_nodes()

        # Step 3: find affinedNumaNode, number of gpus affined to that node
        # and the relative index of the gpu on that node
        affinedNumaNode = numaNodes[self.local_rank]
        affinedGpuCount, affinedGpuIndex = self.get_affined_gpus(numaNodes)

        # Step 4: Get all cpus affined to the numa node
        numaCpuMapVal = self.system.get_numa_node_to_cpus(affinedNumaNode)

        # Step 5: Get CPUs available
        cpuMapVal = self.system.get_available_cpu()

        # Step 6: find available cpus on the numa node
        numaCpus = numaCpuMapVal & cpuMapVal

        # Step 7: find number of cpus
        numaCpuLen = len(bin(numaCpus)) - 2

        # Step 8: get cpu ids on numa node
        cpuList = []
        for i in range(numaCpuLen):
            if (numaCpus >> i) & 1 == 1:
                cpuList.append(i)

        # Step 9: get list of cpus that are fully available
        # considering thread siblings
        coreIDs = {}
        cpuMask = 0
        for i in cpuList:
            if cpuMask & (1 << i) == 0:
                threadsMapVal = self.system.get_thread_siblings(i)
                coreID = (threadsMapVal & (-threadsMapVal)).bit_length() - 1
                if (numaCpus & threadsMapVal) == threadsMapVal:
                    coreIDs[coreID] = threadsMapVal
                cpuMask |= threadsMapVal

        # Step 10: get a list of highest level shared caches in the node
        # sorted by number of available cpus
        cpuMask = 0
        allSharedCacheCpuIDs = []
        cacheCpuCount = {}
        allCacheNames = {}
        for i in cpuList:
            if cpuMask & (1 << i) == 0:
                caches = self.system.get_shared_caches(affinedNumaNode, i)
                # select the highest level cache
                cache = caches[-1]
                cpusInSharedCacheVal = self.system.get_cpus_in_shared_cache(affinedNumaNode, i, cache)
                cpusInSharedCacheValAvailable = cpusInSharedCacheVal & numaCpus
                allSharedCacheCpuIDs.append(i)
                cacheCpuCount[i] = len([ones for ones in bin(cpusInSharedCacheValAvailable)[2:] if ones == '1'])
                allCacheNames[i] = cache
                cpuMask |= cpusInSharedCacheVal
        # stable sort caches by number of cpus available
        allSharedCacheCpuIDs.sort(key=lambda i: -1 * cacheCpuCount[i])

        # Step 11: map gpu to a cache
        affinedCacheID = affinedGpuIndex % len(allSharedCacheCpuIDs)
        cpusSharedCacheVal = self.system.get_cpus_in_shared_cache(affinedNumaNode, allSharedCacheCpuIDs[affinedCacheID],
                                                                  allCacheNames[allSharedCacheCpuIDs[affinedCacheID]])
        cpusSharedCacheVal = cpusSharedCacheVal & numaCpus

        # Step 12.1: create a list of cpus
        resultCpuLen = len(bin(cpusSharedCacheVal)) - 2
        resultCpuList = []
        for i in range(resultCpuLen):
            if (cpusSharedCacheVal >> i) & 1 == 1:
                resultCpuList.append(i)

        # Step 12.2: range-ify the cpu list
        cpu_ranges = self.rangify(resultCpuList)
        numactlargs = self.get_args_string(cpu_ranges, affinedNumaNode)
        return numactlargs


class Socket(Numa):
    """
    implements socket numa-binding
    """

    def __init__(self, local_rank, system):
        super().__init__(local_rank, system)

    def socket_get_nodes_for_rank(self):
        numa_node_list = self.system.get_numa_nodes()
        socket_node = self.system.get_socket_node()
        socket_min = 0
        socket_max = 0
        for key, value in socket_node.items():
            if numa_node_list[self.local_rank] in value:
                socket_min = min(value)
                socket_max = max(value)
        return socket_min, socket_max

    def get_numactl_args(self):
        socket_min, socket_max = self.socket_get_nodes_for_rank()
        numactlargs = []
        if socket_min != socket_max:
            numactlargs += ["--cpunodebind={}-{}".format(socket_min, socket_max),
                            "--membind={}-{}".format(socket_min, socket_max)]
        else:
            numactlargs += ["--cpunodebind={}".format(socket_min), "--membind={}".format(socket_max)]
        return numactlargs


def main():
    args = parse_args()
    current_env = os.environ.copy()
    system = System()

    local_rank, is_rank_required = get_local_rank(args)
    if local_rank == None:
        raise Exception("local_rank is unknown")

    gpu_count = system.get_gpu_count()
    if local_rank >= gpu_count:
        raise Exception(
            "binding not supported, local_rank:{} is greater than gpu count:{}".format(local_rank, gpu_count))

    if args.affinity == "node":
        numa = Node(local_rank, system)
    elif args.affinity == "exclusive":
        numa = Exclusive(local_rank, system)
    elif args.affinity == "core-complex":
        numa = CoreComplex(local_rank, system)
    elif args.affinity == "socket":
        numa = Socket(local_rank, system)
    else:
        raise Exception(f"Unknown affinity: {args.affinity}")

    numactlargs = numa.get_numactl_args()
    cmd = get_cmd(args, numactlargs, local_rank, is_rank_required)
    print(socket.gethostname(), "Local Rank:", local_rank, "Binding:", cmd)
    logger.info("%s.Local Rank:%s Binding:%s", socket.gethostname(), local_rank, cmd)
    run_cmd(cmd, current_env)


if __name__ == "__main__":
    main()
