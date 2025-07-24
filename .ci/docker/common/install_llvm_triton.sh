#!/usr/bin/env bash
set -euo pipefail

# --- Detect platform suffix ---
detect_suffix() {
    local arch=$(uname -m)
    local sys=$(uname -s)

    case "$arch" in
        x86_64) arch="x64" ;;
        aarch64|arm64) arch="arm64" ;;
        *) echo "Unsupported architecture: $arch" >&2; exit 1 ;;
    esac

    if [[ "$sys" == "Darwin" ]]; then
        echo "macos-$arch"
    elif [[ "$sys" == "Linux" ]]; then
        if [[ "$arch" == "arm64" ]]; then
            echo "ubuntu-arm64"
        elif [[ "$arch" == "x64" ]]; then
            local glibc_ver=$(ldd --version | head -n1 | grep -oP '\d+\.\d+')
            local major=$(echo "$glibc_ver" | cut -d. -f1)
            local minor=$(echo "$glibc_ver" | cut -d. -f2)
            local vglibc=$((major * 100 + minor))
            if (( vglibc > 228 )); then
                echo "ubuntu-x64"
            elif (( vglibc > 217 )); then
                echo "almalinux-x64"
            else
                echo "centos-x64"
            fi
        fi
    else
        echo "Unsupported system: $sys" >&2
        exit 1
    fi
}

# --- Fetch LLVM hash ---
get_llvm_hash() {
    local file="/var/lib/jenkins/triton/cmake/llvm-hash.txt"
    if [[ -f "$file" ]]; then
        cut -c1-8 < "$file"
    else
        curl -sSL https://raw.githubusercontent.com/triton-lang/triton/main/cmake/llvm-hash.txt | cut -c1-8
    fi
}

# --- Download and extract LLVM ---
download_and_extract() {
    local url="$1"
    local dest="$2"
    local tarball="${url##*/}"
    local tarpath="$dest/$tarball"

    mkdir -p "$dest"
    curl -fsSL "$url" -o "$tarpath"

    pushd "$dest" >/dev/null
    tar --strip-components=1 -zxf "$tarball"
    rm -f "$tarball"
    popd >/dev/null
}

# --- Main ---
main() {
    local suffix
    suffix=$(detect_suffix)

    local hash
    hash=$(get_llvm_hash)

    local base="llvm-${hash}-${suffix}"
    local url="https://oaitriton.blob.core.windows.net/public/llvm-builds/${base}.tar.gz"
    local install_dir="/opt/llvm"

    download_and_extract "$url" "$install_dir"

    echo ""
    echo "# --- LLVM for Triton ---"
    echo "export PATH=/opt/llvm/bin:\$PATH"
    echo "export LD_LIBRARY_PATH=/opt/llvm/lib:\$LD_LIBRARY_PATH"
    echo "export CMAKE_PREFIX_PATH=/opt/llvm:\$CMAKE_PREFIX_PATH"
    echo "export LLVM_DIR=/opt/llvm"
}

main "$@"
