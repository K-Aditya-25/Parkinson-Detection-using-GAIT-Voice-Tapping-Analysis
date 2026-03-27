"""
Generate a self-signed SSL certificate with proper SAN (Subject Alternative Name).
Safari and iOS require SAN — certs without it are rejected even if you tap 'proceed'.
"""
import socket
import datetime
import ipaddress
import os
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa

CERT_DIR = os.path.join(os.path.dirname(__file__), 'certs')
os.makedirs(CERT_DIR, exist_ok=True)
CERT_FILE = os.path.join(CERT_DIR, 'cert.pem')
KEY_FILE  = os.path.join(CERT_DIR, 'key.pem')


def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return '127.0.0.1'


def generate(local_ip=None):
    if local_ip is None:
        local_ip = get_local_ip()

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, local_ip),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, 'ParkInsight Dev'),
    ])

    now = datetime.datetime.utcnow()
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=365))
        # ← This SAN is what Safari requires
        .add_extension(
            x509.SubjectAlternativeName([
                x509.IPAddress(ipaddress.IPv4Address(local_ip)),
                x509.DNSName('localhost'),
                x509.IPAddress(ipaddress.IPv4Address('127.0.0.1')),
            ]),
            critical=False,
        )
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )

    with open(KEY_FILE, 'wb') as f:
        f.write(key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        ))

    with open(CERT_FILE, 'wb') as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))

    print(f"Certificate generated for IP: {local_ip}")
    print(f"  Cert: {CERT_FILE}")
    print(f"  Key:  {KEY_FILE}")
    print()
    print("=== IMPORTANT: iPhone setup (one-time) ===")
    print(f"1. On your iPhone Safari, open:  https://{local_ip}:5000/cert")
    print(f"   This downloads the certificate file.")
    print(f"2. Go to Settings > General > VPN & Device Management")
    print(f"   Tap the ParkInsight certificate and install it.")
    print(f"3. Go to Settings > General > About > Certificate Trust Settings")
    print(f"   Enable full trust for ParkInsight Dev.")
    print(f"4. Now open https://{local_ip}:5000/phone — it will work!")
    print()
    return CERT_FILE, KEY_FILE


if __name__ == '__main__':
    generate()
