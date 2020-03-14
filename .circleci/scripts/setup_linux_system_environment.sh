#!/usr/bin/env bash
set -eux -o pipefail

# Set up CircleCI GPG keys for apt, if needed
curl --retry 3 -s -L https://packagecloud.io/circleci/trusty/gpgkey | sudo apt-key add -

# Stop background apt updates.  Hypothetically, the kill should not
# be necessary, because stop is supposed to send a kill signal to
# the process, but we've added it for good luck.  Also
# hypothetically, it's supposed to be unnecessary to wait for
# the process to block.  We also have that line for good luck.
# If you like, try deleting them and seeing if it works.
sudo systemctl stop apt-daily.service || true
sudo systemctl kill --kill-who=all apt-daily.service || true

sudo systemctl stop unattended-upgrades.service || true
sudo systemctl kill --kill-who=all unattended-upgrades.service || true

# wait until `apt-get update` has been killed
while systemctl is-active --quiet apt-daily.service
do
    sleep 1;
done
while systemctl is-active --quiet unattended-upgrades.service
do
    sleep 1;
done

# See if we actually were successful
systemctl list-units --all | cat

# For good luck, try even harder to kill apt-get
sudo pkill apt-get || true

# For even better luck, purge unattended-upgrades
sudo apt-get purge -y unattended-upgrades

cat /etc/apt/sources.list

# For the bestest luck, kill again now
sudo pkill apt || true
sudo pkill dpkg || true

# Try to detect if apt/dpkg is stuck
if ps auxfww | grep '[a]pt'; then
  echo "WARNING: There are leftover apt processes; subsequent apt update will likely fail"
fi
if ps auxfww | grep '[d]pkg'; then
  echo "WARNING: There are leftover dpkg processes; subsequent apt update will likely fail"
fi
