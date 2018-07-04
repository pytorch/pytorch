#!/bin/bash

set -ex

# Mirror jenkins user in container
echo "jenkins:x:$JENKINS_UID:$JENKINS_GID::/var/lib/jenkins:" >> /etc/passwd
echo "jenkins:x:$JENKINS_GID:" >> /etc/group

# Create $HOME
mkdir -p /var/lib/jenkins
chown jenkins:jenkins /var/lib/jenkins
mkdir -p /var/lib/jenkins/.ccache
chown jenkins:jenkins /var/lib/jenkins/.ccache

# Allow writing to /usr/local (for make install)
chown jenkins:jenkins /usr/local

# Allow sudo
echo 'jenkins ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/jenkins
