"""
ParkInsight startup script.
- Starts Flask on HTTP port 5000 (dashboard)
- Optionally creates an ngrok HTTPS tunnel (phone gait test on iOS)
"""
import os
import sys
import socket
import threading
import subprocess
import argparse
import webbrowser

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return 'localhost'

def start_flask():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, 'backend')
    from backend.app import app
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ngrok', action='store_true',
                        help='Create HTTPS tunnel via ngrok (needed for iOS gait test)')
    args = parser.parse_args()

    local_ip = get_local_ip()

    print("\n" + "=" * 55)
    print("  ParkInsight")
    print("=" * 55)

    # Start Flask in a background thread
    flask_thread = threading.Thread(target=start_flask, daemon=True)
    flask_thread.start()

    import time
    time.sleep(2)  # Wait for Flask to boot

    if args.ngrok:
        try:
            from pyngrok import ngrok, conf

            # Check for auth token
            token_path = os.path.expanduser('~/.ngrok2/ngrok.yml')
            if not os.path.exists(token_path):
                print("\n  ngrok needs a free auth token (one-time setup):")
                print("  1. Go to https://dashboard.ngrok.com/signup")
                print("  2. Copy your authtoken")
                print("  3. Run:  ngrok config add-authtoken YOUR_TOKEN")
                print("\n  Then re-run:  python start.py --ngrok")
                print()
                # Fall back to HTTP instructions
                args.ngrok = False
            else:
                tunnel = ngrok.connect(5000)
                https_url = tunnel.public_url.replace('http://', 'https://')
                print(f"  Dashboard : http://{local_ip}:5000/dashboard")
                print(f"  Phone URL : {https_url}/phone  (HTTPS — works on iPhone!)")
                print(f"\n  Scan the QR code on the dashboard for the phone URL.")
                print(f"  ngrok tunnel: {https_url}")
                print("=" * 55 + "\n")

                # Update the server-info endpoint to return ngrok URL
                os.environ['PARKINSIGHT_PHONE_URL'] = f"{https_url}/phone"
                os.environ['PARKINSIGHT_BASE_URL'] = https_url

        except Exception as e:
            print(f"  ngrok error: {e}")
            args.ngrok = False

    if not args.ngrok:
        print(f"  Dashboard : http://{local_ip}:5000/dashboard")
        print(f"  Phone URL : http://{local_ip}:5000/phone  (Android/same WiFi)")
        print()
        print("  For iPhone gait test, run with ngrok:")
        print("    python start.py --ngrok")
        print("=" * 55 + "\n")

    # Open dashboard in browser
    webbrowser.open(f'http://localhost:5000/dashboard')

    # Keep alive
    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        if args.ngrok:
            try:
                from pyngrok import ngrok
                ngrok.kill()
            except Exception:
                pass
