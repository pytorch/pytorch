#!/bin/bash
# Helper utilities for build
# Script used only in CD pipeline

OPENSSL_DOWNLOAD_URL=https://www.openssl.org/source/old/1.1.1/
CURL_DOWNLOAD_URL=https://curl.askapache.com/download

AUTOCONF_DOWNLOAD_URL=https://ftp.gnu.org/gnu/autoconf


function check_var {
    if [ -z "$1" ]; then
        echo "required variable not defined"
        exit 1
    fi
}


function do_openssl_build {
    ./config no-ssl2 no-shared -fPIC --prefix=/usr/local/ssl > /dev/null
    make > /dev/null
    make install > /dev/null
}


function check_sha256sum {
    local fname=$1
    check_var ${fname}
    local sha256=$2
    check_var ${sha256}

    echo "${sha256}  ${fname}" > ${fname}.sha256
    sha256sum -c ${fname}.sha256
    rm -f ${fname}.sha256
}


function build_openssl {
    local openssl_fname=$1
    check_var ${openssl_fname}
    local openssl_sha256=$2
    check_var ${openssl_sha256}
    check_var ${OPENSSL_DOWNLOAD_URL}
    curl -sLO ${OPENSSL_DOWNLOAD_URL}/${openssl_fname}.tar.gz
    check_sha256sum ${openssl_fname}.tar.gz ${openssl_sha256}
    tar -xzf ${openssl_fname}.tar.gz
    (cd ${openssl_fname} && do_openssl_build)
    rm -rf ${openssl_fname} ${openssl_fname}.tar.gz
}


function do_curl_build {
    LIBS=-ldl ./configure --with-ssl --disable-shared > /dev/null
    make > /dev/null
    make install > /dev/null
}


function build_curl {
    local curl_fname=$1
    check_var ${curl_fname}
    local curl_sha256=$2
    check_var ${curl_sha256}
    check_var ${CURL_DOWNLOAD_URL}
    curl -sLO ${CURL_DOWNLOAD_URL}/${curl_fname}.tar.bz2
    check_sha256sum ${curl_fname}.tar.bz2 ${curl_sha256}
    tar -jxf ${curl_fname}.tar.bz2
    (cd ${curl_fname} && do_curl_build)
    rm -rf ${curl_fname} ${curl_fname}.tar.bz2
}


function do_standard_install {
    ./configure > /dev/null
    make > /dev/null
    make install > /dev/null
}


function build_autoconf {
    local autoconf_fname=$1
    check_var ${autoconf_fname}
    local autoconf_sha256=$2
    check_var ${autoconf_sha256}
    check_var ${AUTOCONF_DOWNLOAD_URL}
    curl -sLO ${AUTOCONF_DOWNLOAD_URL}/${autoconf_fname}.tar.gz
    check_sha256sum ${autoconf_fname}.tar.gz ${autoconf_sha256}
    tar -zxf ${autoconf_fname}.tar.gz
    (cd ${autoconf_fname} && do_standard_install)
    rm -rf ${autoconf_fname} ${autoconf_fname}.tar.gz
}
