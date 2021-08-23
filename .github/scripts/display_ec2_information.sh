#!/usr/bin/env bash

set -euo pipefail

function get_ec2_metadata() {
    category=$1
    curl -fsSL "http://169.254.169.254/latest/meta-data/${category}"
}

echo "ami-id: $(get_ec2_metadata ami-id)"
echo "instance-id: $(get_ec2_metadata instance-id)"
echo "instance-type: $(get_ec2_metadata instance-type)"
