#!/usr/bin/env bash
# Informational only, never fails: print the raw `amd-smi process` output (and, with --with-monitor,
# the `amd-smi monitor` table), then one line per visible GPU partition. Each amd-smi command runs once.
# occupied: `amd-smi process` lists a process in this pod for the partition.
# in use: no process is listed, but VRAM use is above the 1 GiB idle limit (idle use is 0 to 0.6 GiB by
#   GPU family); the holder may be outside this pod.
# free: a VRAM reading exists and is at or below the limit.
# unknown: no VRAM reading from sysfs or amd-smi. Other pods' processes are invisible to amd-smi,
#   so "no process listed" never proves free.
# MATRIX_RUNNER's `.N` suffix sets the expected device count.

kfd="${KFD_DIR:-/sys/class/kfd/kfd}"
idle_mib=1024
shopt -s nullglob

# Raw output first, so it is in the log even if parsing below goes wrong. -k also kills a call that ignores TERM.
smi_list=""
smi_vram=""
smi_proc=""
proc_ok=""
if command -v amd-smi >/dev/null; then
  smi_list=$(timeout -k 5 30 amd-smi list 2>/dev/null)
  smi_vram=$(timeout -k 5 30 amd-smi monitor -ptumv 2>/dev/null)
  smi_proc=$(timeout -k 5 30 amd-smi process 2>&1) && [[ -n "${smi_proc}" ]] && proc_ok=1
  [[ "${1:-}" != --with-monitor ]] || echo "${smi_vram}"
  echo "${smi_proc}"
fi

# A job pod's own devices are listed here; otherwise use every render node.
devs=()
if [[ -r /etc/podinfo/gha-render-devices ]]; then
  while read -r d; do devs+=("${d}"); done < <(grep -o 'renderD[0-9]*' /etc/podinfo/gha-render-devices)
else
  for d in /dev/dri/renderD*; do devs+=("${d##*/}"); done
fi

summary="GPU partitions: ${#devs[@]} render device(s) visible"
if [[ "${MATRIX_RUNNER:-}" =~ \.([0-9]+)$ ]]; then
  summary+=", expected ${BASH_REMATCH[1]}"
  [[ ${#devs[@]} -eq ${BASH_REMATCH[1]} ]] || summary+=" (MISMATCH)"
fi
echo "${summary}"

for name in "${devs[@]}"; do
  # amd-smi numbers GPUs itself; match on the KFD id both tools report for this render node.
  gpu_id=""
  for props in "${kfd}"/topology/nodes/*/properties; do
    if grep -qx "drm_render_minor ${name#renderD}" "${props}" 2>/dev/null; then
      gpu_id=$(cat "${props%properties}gpu_id" 2>/dev/null)
      break
    fi
  done

  holders=""
  smi_note="amd-smi: unavailable"
  smi_unknown="amd-smi unavailable"
  idx=""
  if [[ -n "${gpu_id}" && "${gpu_id}" != 0 ]]; then
    idx=$(awk -v id="${gpu_id}" '/^GPU:/ {gpu = $2} $1 == "KFD_ID:" && $2 == id {print gpu; exit}' <<<"${smi_list}")
  fi
  if [[ -n "${idx}" && -n "${proc_ok}" ]]; then
    smi_note="amd-smi: no process listed in this pod"
    smi_unknown="amd-smi lists no process in this pod"
    holders=$(awk -v i="${idx}" '$1 == "GPU:" {on = ($2 == i)} on && $1 == "NAME:" {n = $2} on && $1 == "PID:" && $2 ~ /^[0-9]+$/ {printf "%sPID %s %s", sep, $2, n; sep = ", "}' <<<"${smi_proc}")
    [[ -z "${holders}" ]] || smi_note="amd-smi: ${holders}"
  fi

  # VRAM used in MiB: sysfs first, else the VRAM_USED column of `amd-smi monitor` (printed as "<n> MB").
  mib=""
  via=""
  if bytes=$(cat "/sys/class/drm/${name}/device/mem_info_vram_used" 2>/dev/null) && [[ "${bytes}" =~ ^[0-9]+$ ]]; then
    mib=$((bytes >> 20))
  elif [[ -n "${idx}" ]]; then
    mib=$(awk -v i="${idx}" '$1 == i {for (f = 2; f <= NF; f++) if ($f == "MB") {print $(f - 1); exit}}' <<<"${smi_vram}")
    if [[ "${mib}" =~ ^[0-9]+$ ]]; then via=" via amd-smi"; else mib=""; fi
  fi
  vram_note="VRAM unreadable"
  if [[ -n "${mib}" ]]; then
    vram_note=$(awk -v m="${mib}" -v l="${idle_mib}" -v v="${via}" 'BEGIN {printf "VRAM %.1f GiB used%s %s %.1f GiB idle limit", m / 1024, v, (m > l ? ">" : "<="), l / 1024}')
  fi

  if [[ -n "${holders}" ]]; then
    echo "GPU partition ${name}: occupied (${smi_note}; ${vram_note})"
  elif [[ -n "${mib}" ]] && ((mib > idle_mib)); then
    echo "GPU partition ${name}: in use (${smi_note}; ${vram_note})"
  elif [[ -n "${mib}" ]]; then
    echo "GPU partition ${name}: free (${smi_note}; ${vram_note})"
  else
    echo "GPU partition ${name}: unknown (${vram_note}; ${smi_unknown})"
  fi
done
exit 0
