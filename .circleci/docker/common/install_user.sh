#!/bin/bash

set -ex

# Mirror jenkins user in container
# jenkins user as ec2-user should have the same user-id
echo "jenkins:x:1000:1000::/var/lib/jenkins:" >> /etc/passwd
echo "jenkins:x:1000:" >> /etc/group

# Create $HOME
mkdir -p /var/lib/jenkins
chown jenkins:jenkins /var/lib/jenkins
mkdir -p /var/lib/jenkins/.ccache
chown jenkins:jenkins /var/lib/jenkins/.ccache

# Allow writing to /usr/local (for make install)
chown jenkins:jenkins /usr/local

# Allow sudo
# TODO: Maybe we shouldn't
echo 'jenkins ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/jenkins
