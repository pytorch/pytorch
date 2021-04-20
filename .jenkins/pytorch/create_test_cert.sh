#!/bin/bash

TMP_CERT_DIR=$(mktemp -d)

openssl genrsa -out ${TMP_CERT_DIR}/ca.key 2048
openssl req -subj "/C=US/ST=New York/L=New York/O=Gloo Certificate Authority" -new -x509 -days 7300 -key ${TMP_CERT_DIR}/ca.key -sha256 -extensions v3_ca -out ${TMP_CERT_DIR}/ca.pem

openssl genrsa -out ${TMP_CERT_DIR}/pkey.key 2048
openssl req -subj "/C=US/ST=California/L=San Francisco/O=Gloo Testing Company" -sha256 -new -key ${TMP_CERT_DIR}/pkey.key -out ${TMP_CERT_DIR}/csr.csr

openssl x509 -sha256 -req -in ${TMP_CERT_DIR}/csr.csr -CA ${TMP_CERT_DIR}/ca.pem -CAkey ${TMP_CERT_DIR}/ca.key -CAcreateserial -out ${TMP_CERT_DIR}/cert.pem -days 7300

openssl verify -CAfile ${TMP_CERT_DIR}/ca.pem ${TMP_CERT_DIR}/cert.pem

export GLOO_DEVICE_TRANSPORT=TCP_TLS
export GLOO_DEVICE_TRANSPORT_TCP_TLS_PKEY=${TMP_CERT_DIR}/pkey.key
export GLOO_DEVICE_TRANSPORT_TCP_TLS_CERT=${TMP_CERT_DIR}/cert.pem
export GLOO_DEVICE_TRANSPORT_TCP_TLS_CA_FILE=${TMP_CERT_DIR}/ca.pem
