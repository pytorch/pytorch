#!/usr/bin/env bash
# update_alternatives_clang.sh
# chmod u+x update_alternatives_clang.sh
#

update_alternatives() {
    local version=${1}
    local priority=${2}
    local z=${3}
    local slaves=${4}
    local path=${5}
    local cmdln

    cmdln="--verbose --install ${path}${master} ${master} ${path}${master}-${version} ${priority}"
    for slave in ${slaves}; do
        cmdln="${cmdln} --slave ${path}${slave} ${slave} ${path}${slave}-${version}"
    done
    sudo update-alternatives ${cmdln}
}

if [[ ${#} -ne 2 ]]; then
    echo usage: "${0}" clang_version priority
    exit 1
fi

version=${1}
priority=${2}
path="/usr/bin/"

master="llvm-config"
slaves="llvm-addr2line llvm-ar llvm-as llvm-bcanalyzer llvm-bitcode-strip llvm-cat llvm-cfi-verify llvm-cov llvm-c-test llvm-cvtres llvm-cxxdump llvm-cxxfilt llvm-cxxmap llvm-debuginfod llvm-debuginfod-find llvm-diff llvm-dis llvm-dlltool llvm-dwarfdump llvm-dwarfutil llvm-dwp llvm-exegesis llvm-extract llvm-gsymutil llvm-ifs llvm-install-name-tool llvm-jitlink llvm-jitlink-executor llvm-lib llvm-libtool-darwin llvm-link llvm-lipo llvm-lto llvm-lto2 llvm-mc llvm-mca llvm-ml llvm-modextract llvm-mt llvm-nm llvm-objcopy llvm-objdump llvm-omp-device-info llvm-opt-report llvm-otool llvm-pdbutil llvm-PerfectShuffle llvm-profdata llvm-profgen llvm-ranlib llvm-rc llvm-readelf llvm-readobj llvm-reduce llvm-remark-size-diff llvm-rtdyld llvm-sim llvm-size llvm-split llvm-stress llvm-strings llvm-strip llvm-symbolizer llvm-tapi-diff llvm-tblgen llvm-tli-checker llvm-undname llvm-windres llvm-xray"

update_alternatives "${version}" "${priority}" "${master}" "${slaves}" "${path}"

master="clang"
slaves="analyze-build asan_symbolize bugpoint c-index-test clang++ clang-apply-replacements clang-change-namespace clang-check clang-cl clang-cpp clangd clang-doc clang-extdef-mapping clang-format clang-format-diff clang-include-fixer clang-linker-wrapper clang-move clang-nvlink-wrapper clang-offload-bundler clang-offload-packager clang-offload-wrapper clang-pseudo clang-query clang-refactor clang-rename clang-reorder-fields clang-repl clang-scan-deps clang-tidy count diagtool dsymutil FileCheck find-all-symbols git-clang-format hmaptool hwasan_symbolize intercept-build ld64.lld ld.lld llc lld lldb lldb-argdumper lldb-instr lldb-server lldb-vscode lld-link lli lli-child-target modularize not obj2yaml opt pp-trace run-clang-tidy sancov sanstats scan-build scan-build-py scan-view split-file UnicodeNameMappingGenerator verify-uselistorder wasm-ld yaml2obj yaml-bench"

update_alternatives "${version}" "${priority}" "${master}" "${slaves}" "${path}"