#!/usr/bin/env bash
# Informational only, never fails: print whether a process holds each visible GPU partition.
# Reads the KFD sysfs process table (as amd-smi does), so it sees other pods; MATRIX_RUNNER's `.N` suffix sets the expected count.

kfd="${KFD_DIR:-/sys/class/kfd/kfd}"
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

for name in "${devs[@]}"; do
  gpu_id=""
  for props in "${kfd}"/topology/nodes/*/properties; do
    if grep -qx "drm_render_minor ${name#renderD}" "${props}" 2>/dev/null; then
      gpu_id=$(cat "${props%properties}gpu_id" 2>/dev/null)
      break
    fi
  done
  if [[ -z "${gpu_id}" || "${gpu_id}" == 0 ]]; then
    echo "GPU partition ${name}: unknown (no KFD node for this render device)"
    continue
  fi

  procs=("${kfd}"/proc/[0-9]*)
  holders=()
  readable=0
  for p in "${procs[@]}"; do
    vram=$(cat "${p}/vram_${gpu_id}" 2>/dev/null) || continue
    readable=1
    [[ "${vram}" =~ ^[0-9]+$ && "${vram}" -gt 0 ]] || continue
    holders+=("PID ${p##*/} $(cat "/proc/${p##*/}/comm" 2>/dev/null || echo '?')")
  done

  used="VRAM unknown"
  if bytes=$(cat "/sys/class/drm/${name}/device/mem_info_vram_used" 2>/dev/null) && [[ "${bytes}" =~ ^[0-9]+$ ]]; then
    used=$(awk -v b="${bytes}" 'BEGIN {printf "VRAM %.1f GiB used", b / 2^30}')
  fi

  if [[ ! -d "${kfd}/proc" || ( ${#procs[@]} -gt 0 && ${readable} -eq 0 ) ]]; then
    echo "GPU partition ${name}: unknown (KFD does not expose per-process VRAM; ${used})"
  elif [[ ${#holders[@]} -gt 0 ]]; then
    list=$(printf '%s, ' "${holders[@]}")
    echo "GPU partition ${name}: occupied (${list}${used})"
  else
    echo "GPU partition ${name}: free (${used})"
  fi
done
exit 0
