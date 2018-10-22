#!/bin/bash

set -ex

[ -n "$CENTOS_VERSION" ]

yum install -y centos-release-scl
yum install -y devtoolset-$CENTOS_VERSION

echo "source scl_source enable devtoolset-$CENTOS_VERSION" >> /root/.bashrc
