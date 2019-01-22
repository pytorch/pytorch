#!/bin/bash

set -ex

# Mirror jenkins user in container
groupadd -g 1014 jenkins
useradd -u 1014 -g 1014 -d /var/lib/jenkins -m jenkins

chown jenkins:jenkins /var/lib/jenkins
mkdir -p /var/lib/jenkins/.ccache
chown jenkins:jenkins /var/lib/jenkins/.ccache

# Allow writing to /usr/local (for make install)
chown jenkins:jenkins /usr/local

# Allow sudo
echo 'jenkins ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/jenkins
