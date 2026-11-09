#!/usr/bin/env bash
# Informational only, never fails: print whether each visible GPU partition is occupied.
# occupied: `amd-smi process` lists a process in this pod, or VRAM use is above the 1 GiB idle limit
#   (idle use is 0 to 0.6 GiB by GPU family). free: a VRAM reading exists and is at or below the limit.
#   unknown: no VRAM reading from sysfs or amd-smi. Other pods' processes are invisible to amd-smi,
#   so "no process listed" never proves free.
# MATRIX_RUNNER's `.N` suffix sets the expected device count.

kfd="${KFD_DIR:-/sys/class/kfd/kfd}"
idle_mib=1024
shopt -s nullglob

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

smi_list=""
smi_vram=""
if command -v amd-smi >/dev/null; then
  smi_list=$(timeout 30 amd-smi list 2>/dev/null)
  smi_vram=$(timeout 30 amd-smi monitor -ptumv 2>/dev/null)
fi

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
  if [[ -n "${idx}" ]] && out=$(timeout 30 amd-smi process --gpu "${idx}" 2>/dev/null) && [[ -n "${out}" ]]; then
    smi_note="amd-smi: no process listed in this pod"
    smi_unknown="amd-smi lists no process in this pod"
    holders=$(awk '$1 == "NAME:" {n = $2} $1 == "PID:" && $2 ~ /^[0-9]+$/ {printf "%sPID %s %s", sep, $2, n; sep = ", "}' <<<"${out}")
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

  if [[ -n "${holders}" || ( -n "${mib}" && ${mib} -gt ${idle_mib} ) ]]; then
    echo "GPU partition ${name}: occupied (${smi_note}; ${vram_note})"
  elif [[ -n "${mib}" ]]; then
    echo "GPU partition ${name}: free (${smi_note}; ${vram_note})"
  else
    echo "GPU partition ${name}: unknown (${vram_note}; ${smi_unknown})"
  fi
done
exit 0
