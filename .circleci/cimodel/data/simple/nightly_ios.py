import cimodel.data.simple.ios_definitions as ios_definitions


class IOSNightlyJob:
    def __init__(self,
                 variant,
                 is_upload=False):

        self.variant = variant
        self.is_upload = is_upload

    def get_phase_name(self):
        return "upload" if self.is_upload else "build"

    def get_common_name_pieces(self, with_version_dots):

        extra_name_suffix = [self.get_phase_name()] if self.is_upload else []

        common_name_pieces = [
            "ios",
        ] + ios_definitions.XCODE_VERSION.render_dots_or_parts(with_version_dots) + [
            "nightly",
            self.variant,
            "build",
        ] + extra_name_suffix

        return common_name_pieces

    def gen_job_name(self):
        return "_".join(["pytorch"] + self.get_common_name_pieces(False))

    def gen_tree(self):
        extra_requires = [x.gen_job_name() for x in BUILD_CONFIGS] if self.is_upload else []

        props_dict = {
            "build_environment": "-".join(["libtorch"] + self.get_common_name_pieces(True)),
            "requires": extra_requires,
            "context": "org-member",
            "filters": {"branches": {"only": "nightly"}},
        }

        if not self.is_upload:
            props_dict["ios_arch"] = self.variant
            props_dict["ios_platform"] = ios_definitions.get_platform(self.variant)
            props_dict["name"] = self.gen_job_name()

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


WORKFLOW_DATA = BUILD_CONFIGS + [
    IOSNightlyJob("binary", is_upload=True),
]


def get_workflow_jobs():
    return [item.gen_tree() for item in WORKFLOW_DATA]
