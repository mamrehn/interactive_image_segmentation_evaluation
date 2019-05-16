from functools import wraps
from socket import gethostname
from pathlib import Path

from OpenSSL import crypto
from flask import Flask, request, Response, send_from_directory


def create_self_signed_certificate(certificate_directory, pub_name, cert_name, priv_name):

    certificate_directory = Path(certificate_directory)
    certificate_directory.mkdir(parents=True, exist_ok=True)
    pub_file = certificate_directory.joinpath(pub_name)
    cert_file = certificate_directory.joinpath(cert_name)
    priv_file = certificate_directory.joinpath(priv_name)

    if pub_file.is_file() and cert_file.is_file() and priv_file.is_file():
        return

    # Create a new key pair
    key = crypto.PKey()
    key.generate_key(type=crypto.TYPE_RSA, bits=2048)

    # Create a self-signed certificate
    cert = crypto.X509()
    cert.get_subject().C = 'DE'
    cert.get_subject().ST = 'Bayern'
    cert.get_subject().L = 'Erlangen'
    cert.get_subject().O = 'Friedrich-Alexander-Universitaet Erlangen-Nuernberg'
    cert.get_subject().OU = "Pattern Recognition Lab"
    cert.get_subject().CN = gethostname()
    cert.set_serial_number(1000)
    cert.gmtime_adj_notBefore(0)
    cert.gmtime_adj_notAfter(10 * 365 * 24 * 60 * 60)
    cert.set_issuer(cert.get_subject())
    cert.set_pubkey(key)
    # https://www.ibm.com/developerworks/community/blogs/SterlingB2B/entry/Why_use_SHA256_instead_of_SHA1?lang=en
    cert.sign(key, digest='sha256')

    pub_file.write_bytes(crypto.dump_publickey(crypto.FILETYPE_PEM, key))
    cert_file.write_bytes(crypto.dump_certificate(crypto.FILETYPE_PEM, cert))
    priv_file.write_bytes(crypto.dump_privatekey(crypto.FILETYPE_PEM, key))


def check_auth(username, password):
    """This function is called to check if a username/password combination is valid."""
    return 'lme' == username and 'lme' == password


def authenticate():
    """Sends a 401 response that enables basic auth"""
    msg = 'Could not verify your access level for that URL.\nYou have to login with proper credentials'
    return Response(msg, 401, {'WWW-Authenticate': 'Basic realm="Start Segmentation"'})


def requires_auth(f):

    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)

    return decorated


def init_server(root_directory_='webapp/deploy'):
    if not Path(root_directory_).is_dir() or 1 > sum(1 for p in Path(root_directory_).iterdir() if p.is_file()):
        root_directory_ = 'webapp'
    print('Root directory is "./{}/"'.format(root_directory_))

    # Set the project root directory as the static folder, you can set others.
    app_ = Flask(__name__, static_folder=root_directory_)
    # os.path.join('js', path).replace('\\','/'))
    return app_, root_directory_


app, root_directory = init_server()


@app.route('/<path:path>')
@requires_auth
def request_page(path):
    return send_from_directory(root_directory, path)


@app.route('/')
@requires_auth
def send_webapp_project_root_file():
    return send_from_directory(root_directory, 'index.html')


if '__main__' == __name__:
    create_self_signed_certificate(certificate_directory='.auth_keys', pub_name='mykeyforflaskauth.pub',
                                   cert_name='mykeyforflaskauth.crt', priv_name='mykeyforflaskauth.key')
    context = ('.auth_keys/mykeyforflaskauth.crt', '.auth_keys/mykeyforflaskauth.key')

    print('\n***********************************************************************************************')
    print('* There might be a warning in your browser similar to "MOZILLA_PKIX_ERROR_SELF_SIGNED_CERT"   *')
    print('* This is expected, since this script uses a self-signed certificate for authentication.      *')
    print('* The warning can therefore be safely ignored.                                                *')
    print('* Better yet, use a service like https://letsencrypt.org/ to sign your certificates.          *')
    print('***********************************************************************************************\n')

    app.run(port=1234, debug=False, ssl_context=context)
