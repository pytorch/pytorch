from datetime import datetime, timedelta
from tempfile import mkdtemp
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes

temp_dir = mkdtemp()
print(temp_dir)


def genrsa(path):
    key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )
    with open(path, "wb") as f:
        f.write(key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        ))
    return key


def create_cert(path, C, ST, L, O, key):
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, C),
        x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, ST),
        x509.NameAttribute(NameOID.LOCALITY_NAME, L),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, O),
    ])
    cert = x509.CertificateBuilder().subject_name(
        subject
    ).issuer_name(
        issuer
    ).public_key(
        key.public_key()
    ).serial_number(
        x509.random_serial_number()
    ).not_valid_before(
        datetime.utcnow()
    ).not_valid_after(
        # Our certificate will be valid for 10 days
        datetime.utcnow() + timedelta(days=10)
    ).add_extension(
        x509.BasicConstraints(ca=True, path_length=None), critical=True,
    ).sign(key, hashes.SHA256())
    # Write our certificate out to disk.
    with open(path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    return cert


def create_req(path, C, ST, L, O, key):
    csr = x509.CertificateSigningRequestBuilder().subject_name(x509.Name([
        # Provide various details about who we are.
        x509.NameAttribute(NameOID.COUNTRY_NAME, C),
        x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, ST),
        x509.NameAttribute(NameOID.LOCALITY_NAME, L),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, O),
    ])).sign(key, hashes.SHA256())
    with open(path, "wb") as f:
        f.write(csr.public_bytes(serialization.Encoding.PEM))
    return csr


def sign_certificate_request(path, csr_cert, ca_cert, private_ca_key):
    cert = x509.CertificateBuilder().subject_name(
        csr_cert.subject
    ).issuer_name(
        ca_cert.subject
    ).public_key(
        csr_cert.public_key()
    ).serial_number(
        x509.random_serial_number()
    ).not_valid_before(
        datetime.utcnow()
    ).not_valid_after(
        # Our certificate will be valid for 10 days
        datetime.utcnow() + timedelta(days=10)
        # Sign our certificate with our private key
    ).sign(private_ca_key, hashes.SHA256())
    with open(path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    return cert


ca_key = genrsa(temp_dir + "/ca.key")
ca_cert = create_cert(temp_dir + "/ca.pem", "US", "New York", "New York", "Gloo Certificate Authority", ca_key)

pkey = genrsa(temp_dir + "/pkey.key")
csr = create_req(temp_dir + "/csr.csr", "US", "California", "San Francisco", "Gloo Testing Company", pkey)

cert = sign_certificate_request(temp_dir + "/cert.pem", csr, ca_cert, ca_key)
