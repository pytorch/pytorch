import cimodel.data.simple.util.branch_filters
import cimodel.lib.miniutils as miniutils
from cimodel.data.simple.util.versions import CudaVersion


class WindowsJob:
    def __init__(
        self,
        test_index,
        vscode_spec,
        cuda_version,
        force_on_cpu=False,
        master_only_pred=lambda job: job.vscode_spec.year != 2019,
    ):
        self.test_index = test_index
        self.vscode_spec = vscode_spec
        self.cuda_version = cuda_version
        self.force_on_cpu = force_on_cpu
        self.master_only_pred = master_only_pred

    def gen_tree(self):

        base_phase = "build" if self.test_index is None else "test"
        numbered_phase = (
            base_phase if self.test_index is None else base_phase + str(self.test_index)
        )

        key_name = "_".join(["pytorch", "windows", base_phase])

        cpu_forcing_name_parts = ["on", "cpu"] if self.force_on_cpu else []

        target_arch = self.cuda_version.render_dots() if self.cuda_version else "cpu"

        base_name_parts = [
            "pytorch",
            "windows",
            self.vscode_spec.render(),
            "py36",
            target_arch,
        ]

        prerequisite_jobs = []
        if base_phase == "test":
            prerequisite_jobs.append("_".join(base_name_parts + ["build"]))

        if self.cuda_version:
            self.cudnn_version = 8 if self.cuda_version.major == 11 else 7

        arch_env_elements = (
            ["cuda" + str(self.cuda_version.major), "cudnn" + str(self.cudnn_version)]
            if self.cuda_version
            else ["cpu"]
        )

        build_environment_string = "-".join(
            ["pytorch", "win"]
            + self.vscode_spec.get_elements()
            + arch_env_elements
            + ["py3"]
        )

        is_running_on_cuda = bool(self.cuda_version) and not self.force_on_cpu

        props_dict = {
            "build_environment": build_environment_string,
            "python_version": miniutils.quote("3.6"),
            "vc_version": miniutils.quote(self.vscode_spec.dotted_version()),
            "vc_year": miniutils.quote(str(self.vscode_spec.year)),
            "vc_product": self.vscode_spec.get_product(),
            "use_cuda": miniutils.quote(str(int(is_running_on_cuda))),
            "requires": prerequisite_jobs,
        }

        if self.master_only_pred(self):
            props_dict[
                "filters"
            ] = cimodel.data.simple.util.branch_filters.gen_filter_dict()

        name_parts = base_name_parts + cpu_forcing_name_parts + [numbered_phase]

        if base_phase == "test":
            test_name = "-".join(["pytorch", "windows", numbered_phase])
            props_dict["test_name"] = test_name

            if is_running_on_cuda:
                props_dict["executor"] = "windows-with-nvidia-gpu"

        props_dict["cuda_version"] = (
            miniutils.quote(str(self.cuda_version))
            if self.cuda_version
            else "cpu"
        )

        props_dict["name"] = "_".join(name_parts)

        return [{key_name: props_dict}]


class VcSpec:
    def __init__(self, year, version_elements=None, hide_version=False):
        self.year = year
        self.version_elements = version_elements or []
        self.hide_version = hide_version

    def get_elements(self):
        if self.hide_version:
            return [self.prefixed_year()]
        return [self.prefixed_year()] + self.version_elements

    def get_product(self):
        return "Community" if self.year == 2019 else "BuildTools"

    def dotted_version(self):
        return ".".join(self.version_elements)

    def prefixed_year(self):
        return "vs" + str(self.year)

    def render(self):
        return "_".join(self.get_elements())

def FalsePred(_):
    return False

def TruePred(_):
    return True

_VC2019 = VcSpec(2019)

WORKFLOW_DATA = [
    # VS2019 CUDA-10.1
    WindowsJob(None, _VC2019, CudaVersion(10, 1)),
    WindowsJob(1, _VC2019, CudaVersion(10, 1)),
    WindowsJob(2, _VC2019, CudaVersion(10, 1)),
    # VS2019 CUDA-11.1
    WindowsJob(None, _VC2019, CudaVersion(11, 1)),
    WindowsJob(1, _VC2019, CudaVersion(11, 1), master_only_pred=TruePred),
    WindowsJob(2, _VC2019, CudaVersion(11, 1), master_only_pred=TruePred),
    # VS2019 CPU-only
    WindowsJob(None, _VC2019, None),
    WindowsJob(1, _VC2019, None, master_only_pred=TruePred),
    WindowsJob(2, _VC2019, None, master_only_pred=TruePred),
    WindowsJob(1, _VC2019, CudaVersion(10, 1), force_on_cpu=True, master_only_pred=TruePred),
]


def get_windows_workflows():
    return [item.gen_tree() for item in WORKFLOW_DATA]
