#!/bin/bash

set -ex

# Mirror jenkins user in container
echo "jenkins:x:1001:1001::/var/lib/jenkins:" >> /etc/passwd
echo "jenkins:x:1001:" >> /etc/group

# Create $HOME
mkdir -p /var/lib/jenkins
chown jenkins:jenkins /var/lib/jenkins

# Allow writing to /usr/local (for make install)
chown jenkins:jenkins /usr/local

# Allow sudo
echo 'jenkins ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/jenkins
