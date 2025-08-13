#!/bin/bash

CREATE_TEST_CERT="$(dirname "${BASH_SOURCE[0]}")/create_test_cert.py"
TMP_CERT_DIR=$(python "$CREATE_TEST_CERT")

openssl verify -CAfile "${TMP_CERT_DIR}/ca.pem" "${TMP_CERT_DIR}/cert.pem"

export GLOO_DEVICE_TRANSPORT=TCP_TLS
export GLOO_DEVICE_TRANSPORT_TCP_TLS_PKEY=${TMP_CERT_DIR}/pkey.key
export GLOO_DEVICE_TRANSPORT_TCP_TLS_CERT=${TMP_CERT_DIR}/cert.pem
export GLOO_DEVICE_TRANSPORT_TCP_TLS_CA_FILE=${TMP_CERT_DIR}/ca.pem

time python test/run_test.py --include distributed/test_c10d_gloo --verbose -- ProcessGroupGlooTest

unset GLOO_DEVICE_TRANSPORT
unset GLOO_DEVICE_TRANSPORT_TCP_TLS_PKEY
unset GLOO_DEVICE_TRANSPORT_TCP_TLS_CERT
unset GLOO_DEVICE_TRANSPORT_TCP_TLS_CA_FILE
