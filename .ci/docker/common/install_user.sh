#!/bin/bash

set -ex

# Since version 24 the system ships with user 'ubuntu' that has id 1000
# We need a work-around to enable id 1000 usage for this script
if [[ $UBUNTU_VERSION == 24.04 ]]; then
    # touch is used to disable harmless error message
    touch /var/mail/ubuntu && chown ubuntu /var/mail/ubuntu && userdel -r ubuntu
fi

# Mirror jenkins user in container
# jenkins user as ec2-user should have the same user-id
echo "jenkins:x:1000:1000::/var/lib/jenkins:" >> /etc/passwd
echo "jenkins:x:1000:" >> /etc/group
# Needed on focal or newer
echo "jenkins:*:19110:0:99999:7:::" >>/etc/shadow

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

# Work around bug where devtoolset replaces sudo and breaks it.
if [ -n "$DEVTOOLSET_VERSION" ]; then
  SUDO=/bin/sudo
else
  SUDO=sudo
fi

# Test that sudo works
$SUDO -u jenkins $SUDO -v
