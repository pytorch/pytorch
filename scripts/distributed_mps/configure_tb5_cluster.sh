#!/usr/bin/env bash
# Configure every Mac in a TB-connected cluster for JACCL RDMA: destroy
# bridge0, create a dedicated "tb5" network location with a DHCP/APIPA
# service per Thunderbolt port, and print each node's per-port IPs.
#
# Usage: ./configure_tb5_cluster.sh <user@host> [user@host ...]
#
# Requires: key-based SSH to each peer, sudo on every node, and `rdma_ctl
# enable` already run once per Mac from macOS Recovery.

set -euo pipefail

usage() {
  cat >&2 <<EOF
Usage: $0 <peer> [peer ...]
  each peer is user@host, reachable via SSH.
EOF
  exit 1
}

[[ $# -lt 1 ]] && usage

CONFIG_SCRIPT=$(cat <<'CONFIG_EOF'
#!/usr/bin/env bash
set -euo pipefail

PREFS="/Library/Preferences/SystemConfiguration/preferences.plist"
LOCATION="tb5"

ifconfig bridge0 &>/dev/null && {
  ifconfig bridge0 | grep -q 'member' && {
    ifconfig bridge0 | awk '/member/ {print $2}' | xargs -n1 ifconfig bridge0 deletem 2>/dev/null || true
  }
  ifconfig bridge0 destroy 2>/dev/null || true
}

/usr/libexec/PlistBuddy -c "Delete :VirtualNetworkInterfaces:Bridge:bridge0" "$PREFS" 2>/dev/null || true

networksetup -listlocations | grep -q "^${LOCATION}$" || {
  networksetup -createlocation "$LOCATION"
}

networksetup -switchtolocation "$LOCATION"
networksetup -listallhardwareports |
  awk -F': ' '/Hardware Port: / {print $2}' |
  while IFS=":" read -r name; do
    case "$name" in
    "Ethernet Adapter"*) ;;
    "Thunderbolt Bridge") ;;
    "Thunderbolt "*)
      num="${name##Thunderbolt }"
      svc="TB5-${num}"
      networksetup -listallnetworkservices |
        grep -q "^${svc}$" ||
        networksetup -createnetworkservice "$svc" "$name" 2>/dev/null ||
        continue
      networksetup -setdhcp "$svc"
      ;;
    *)
      networksetup -listallnetworkservices |
        grep -q "^${name}$" ||
        networksetup -createnetworkservice "$name" "$name" 2>/dev/null ||
        continue
      ;;
    esac
  done

networksetup -listnetworkservices | grep -q "Thunderbolt Bridge" && {
  networksetup -setnetworkserviceenabled "Thunderbolt Bridge" off
} || true
CONFIG_EOF
)

REPORT_CMD=$'for i in $(ifconfig -l | tr " " "\\n" | grep -E "^en[0-9]+$"); do ip=$(ifconfig "$i" 2>/dev/null | awk "/inet /{print \\$2}"); [[ -n "$ip" ]] && echo "  $i  $ip"; done'

TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT
printf '%s\n' "$CONFIG_SCRIPT" > "$TMP"
chmod +x "$TMP"

echo "==> configuring local machine"
sudo bash "$TMP"

for peer in "$@"; do
  echo "==> configuring $peer"
  scp -q "$TMP" "${peer}:/tmp/tb5_config.sh"
  ssh -t "$peer" "sudo bash /tmp/tb5_config.sh && rm -f /tmp/tb5_config.sh"
done

sleep 3

echo
echo "==> per-port APIPA IPs (pick any live one as MASTER_ADDR)"
echo "local:"
bash -c "$REPORT_CMD"

for peer in "$@"; do
  echo "$peer:"
  ssh "$peer" "bash -c '$REPORT_CMD'"
done

cat <<'EOF'

Next steps:
  - pick one node to be rank 0; note one of its TB-port IPs as MASTER_ADDR
  - on every rank: export MASTER_ADDR=<IP> MASTER_PORT=29501, then
        dist.init_process_group(backend="mps", rank=<k>, world_size=<N>)
  - see docs/source/distributed.mps.md for a full example
EOF
