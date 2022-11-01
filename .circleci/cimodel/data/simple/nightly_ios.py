import cimodel.data.simple.ios_definitions as ios_definitions
import cimodel.lib.miniutils as miniutils


class IOSNightlyJob:
    def __init__(self,
                 variant,
                 is_full_jit=False,
                 is_upload=False):

        self.variant = variant
        self.is_full_jit = is_full_jit
        self.is_upload = is_upload

    def get_phase_name(self):
        return "upload" if self.is_upload else "build"

    def get_common_name_pieces(self, sep):

        extra_name_suffix = [self.get_phase_name()] if self.is_upload else []

        extra_name = ["full_jit"] if self.is_full_jit else []

        common_name_pieces = [
            "ios",
        ] + extra_name + [
        ] + ios_definitions.XCODE_VERSION.render_dots_or_parts(sep) + [
            "nightly",
            self.variant,
            "build",
        ] + extra_name_suffix

        return common_name_pieces

    def gen_job_name(self):
        return "_".join(["pytorch"] + self.get_common_name_pieces(None))

    def gen_tree(self):
        build_configs = BUILD_CONFIGS_FULL_JIT if self.is_full_jit else BUILD_CONFIGS
        extra_requires = [x.gen_job_name() for x in build_configs] if self.is_upload else []

        props_dict = {
            "build_environment": "-".join(["libtorch"] + self.get_common_name_pieces(".")),
            "requires": extra_requires,
            "context": "org-member",
            #"filters": {"branches": {"only": "nightly"}},
        }

        if not self.is_upload:
            props_dict["ios_arch"] = self.variant
            props_dict["ios_platform"] = ios_definitions.get_platform(self.variant)
            props_dict["name"] = self.gen_job_name()
            props_dict["use_metal"] = miniutils.quote(str(int(True)))
            props_dict["use_coreml"] = miniutils.quote(str(int(True)))

        if self.is_full_jit:
            props_dict["lite_interpreter"] = miniutils.quote(str(int(False)))

        props_dict["use_metal"] = miniutils.quote(str(int(False)))
        props_dict["use_coreml"] = miniutils.quote(str(int(False)))

        template_name = "_".join([
            "binary",
            "ios",
            self.get_phase_name(),
        ])

        return [{template_name: props_dict}]


BUILD_CONFIGS = [
    IOSNightlyJob("x86_64"),
    IOSNightlyJob("arm64"),
]

BUILD_CONFIGS_FULL_JIT = [
    IOSNightlyJob("x86_64", is_full_jit=True),
    IOSNightlyJob("arm64", is_full_jit=True),
]

WORKFLOW_DATA = BUILD_CONFIGS + BUILD_CONFIGS_FULL_JIT + [
    IOSNightlyJob("binary", is_full_jit=False, is_upload=True),
    IOSNightlyJob("binary", is_full_jit=True, is_upload=True),
]


def get_workflow_jobs():
    return [item.gen_tree() for item in WORKFLOW_DATA]
