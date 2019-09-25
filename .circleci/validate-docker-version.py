#!/usr/bin/env python3

import urllib.request
import re

import cimodel.data.pytorch_build_definitions as pytorch_build_definitions
import cimodel.data.caffe2_build_definitions as caffe2_build_definitions

RE_VERSION = re.compile(r'allDeployedVersions = "([0-9,]+)"')

URL_TEMPLATE = (
    "https://raw.githubusercontent.com/pytorch/ossci-job-dsl/"
    "master/src/main/groovy/ossci/{}/DockerVersion.groovy"
)

def check_version(job, expected_version):
    url = URL_TEMPLATE.format(job)
    with urllib.request.urlopen(url) as f:
        contents = f.read().decode('utf-8')
        m = RE_VERSION.search(contents)
        if not m:
            raise RuntimeError(
                "Unbelievable! I could not find the variable allDeployedVersions in "
                "{}; did the organization of ossci-job-dsl change?\n\nFull contents:\n{}"
                .format(url, contents)
            )
        valid_versions = [int(v) for v in m.group(1).split(',')]
        if expected_version not in valid_versions:
            raise RuntimeError(
                "We configured {} to use Docker version {}; but this "
                "version is not deployed in {}.  Non-deployed versions will be "
                "garbage collected two weeks after they are created.  DO NOT LAND "
                "THIS TO MASTER without also updating ossci-job-dsl with this version."
                "\n\nDeployed versions: {}"
                .format(job, expected_version, url, m.group(1))
            )

def validate_docker_version():
    check_version('pytorch', pytorch_build_definitions.DOCKER_IMAGE_VERSION)
    check_version('caffe2', caffe2_build_definitions.DOCKER_IMAGE_VERSION)


if __name__ == "__main__":
    validate_docker_version()
