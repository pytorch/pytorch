#!/usr/bin/env bash

grab_prune_version() {
    conda search -c "${CHANNEL}" --platform "${platform}" "${PKG}" 2>/dev/null | \
        grep "${CHANNEL}" | \
        awk -F '  *' '{print $2}' | \
        uniq | \
        head -n -1 | \
        xargs
}

grab_latest_version() {
    conda search -c "${CHANNEL}" --platform "${platform}" "${PKG}" 2>/dev/null | \
        grep "${CHANNEL}" | \
        awk -F '  *' '{print $2}' | \
        uniq | \
        tail -n 1 | \
        xargs
}

grab_specs_for_version() {
    conda search -c "${CHANNEL}" --platform "${platform}" "${PKG}" 2>/dev/null | \
        grep "${CHANNEL}" | \
        grep "$1" | \
        awk -F '  *' '{print $3}' | \
        uniq | \
        xargs
}

set -eou pipefail

CHANNEL=${CHANNEL:-pytorch-nightly}
PKG=${PKG:-pytorch}
PLATFORMS=${PLATFORMS:-noarch osx-64 osx-arm64 linux-64 win-64}

for platform in ${PLATFORMS}; do
    latest_version="$(grab_latest_version || true)"
    specs_in_latest_version="$(grab_specs_for_version "${latest_version}" || true)"
    versions_to_prune="$(grab_prune_version || true)"
    for version in ${versions_to_prune}; do
        specs_in_prune_version="$(grab_specs_for_version "${version}" || true)"
        for spec in ${specs_in_prune_version}; do
        # If this spec is included in specs_in_latest_version, then remove it.
        if [[ "${specs_in_latest_version}" =~ ${spec} ]];then
            (
                set -x
                anaconda remove --force "${CHANNEL}/${PKG}/${version}/${platform}/${PKG}-${version}-${spec}.tar.bz2"
            )
        fi
        done
    done
done
