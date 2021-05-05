#!/bin/bash

TMP_CERT_DIR=$(python create_test_cert.py)

openssl verify -CAfile "${TMP_CERT_DIR}/ca.pem" "${TMP_CERT_DIR}/cert.pem"

export GLOO_DEVICE_TRANSPORT=TCP_TLS
export GLOO_DEVICE_TRANSPORT_TCP_TLS_PKEY=${TMP_CERT_DIR}/pkey.key
export GLOO_DEVICE_TRANSPORT_TCP_TLS_CERT=${TMP_CERT_DIR}/cert.pem
export GLOO_DEVICE_TRANSPORT_TCP_TLS_CA_FILE=${TMP_CERT_DIR}/ca.pem
