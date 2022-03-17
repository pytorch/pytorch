#!/usr/bin/env bash

set -euo pipefail
function get_ec2_metadata() {
    # Pulled from instance metadata endpoint for EC2
    # see https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/instancedata-data-retrieval.html
    category=$1
    curl -fsSL "http://169.254.169.254/latest/meta-data/${category}"
}
echo "ami-id: $(get_ec2_metadata ami-id)"
echo "instance-id: $(get_ec2_metadata instance-id)"
echo "instance-type: $(get_ec2_metadata instance-type)"
echo "system info $(uname -a)"
