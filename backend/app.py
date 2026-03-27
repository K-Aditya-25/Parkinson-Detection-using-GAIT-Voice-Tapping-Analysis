"""
ParkInsight Flask Backend
Serves API endpoints for gait analysis, tapping analysis, voice analysis.
Serves phone capture page and dashboard.
"""
import os
import json
import io
import csv
import socket
import numpy as np
import joblib
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Import feature extraction
import sys
sys.path.insert(0, os.path.dirname(__file__))
from feature_extraction.gait_features import extract_gait_features, get_feature_names
from feature_extraction.tapping_features import extract_tapping_features, assess_tapping_risk

app = Flask(__name__)
CORS(app)

# ---- Load Models ----
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')

try:
    gait_bundle = joblib.load(os.path.join(MODEL_DIR, 'gait_model.pkl'))
    gait_model = gait_bundle['model']
    gait_scaler = gait_bundle['scaler']
    gait_feature_names = gait_bundle['feature_names']
    print(f"Gait model loaded ({len(gait_feature_names)} features)")
except Exception as e:
    print(f"Warning: Could not load gait model: {e}")
    gait_model = None
    gait_scaler = None
    gait_feature_names = get_feature_names()

try:
    voice_bundle = joblib.load(os.path.join(MODEL_DIR, 'voice_model.pkl'))
    voice_model = voice_bundle['model']
    voice_scaler = voice_bundle['scaler']
    voice_feature_names = voice_bundle['feature_names']
    print(f"Voice model loaded ({len(voice_feature_names)} features)")
except Exception as e:
    print(f"Warning: Could not load voice model: {e}")
    voice_model = None
    voice_scaler = None
    voice_feature_names = []

# ---- Store latest results in memory ----
latest_results = {
    'gait': None,
    'tapping': None,
    'voice': None,
    'combined': None
}

# ---- Load demo data ----
DEMO_DIR = os.path.join(os.path.dirname(__file__), 'demo_data')


def load_demo(scenario):
    filepath = os.path.join(DEMO_DIR, f'{scenario}.json')
    if os.path.exists(filepath):
        with open(filepath) as f:
            return json.load(f)
    return None


# ======== SHAP (lazy import due to cv2 issue) ========
def compute_shap_values(model, feature_scaled, feature_names):
    """Compute SHAP values, handling import issues gracefully."""
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(feature_scaled)
        if isinstance(shap_vals, list):
            # Binary classification: shap_vals[1] is for positive class
            vals = shap_vals[1][0]
        else:
            vals = shap_vals[0]
        return dict(zip(feature_names, [float(v) for v in vals]))
    except Exception as e:
        print(f"SHAP computation failed: {e}")
        # Return dummy SHAP values based on feature importance
        try:
            importances = model.feature_importances_
            return dict(zip(feature_names, [float(v) for v in importances]))
        except:
            return {name: 0.0 for name in feature_names}


# ======== API ENDPOINTS ========

@app.route('/api/gait/analyze', methods=['POST'])
def analyze_gait():
    """Receive raw accelerometer JSON from phone, extract features, predict."""
    try:
        data = request.json.get('sensor_data', [])

        if len(data) < 50:
            return jsonify({'error': 'Not enough sensor data. Walk for at least 10 seconds.'}), 400

        # Convert to numpy arrays
        timestamps = np.array([d['t'] for d in data])
        acc_x = np.array([d['ax'] for d in data])
        acc_y = np.array([d['ay'] for d in data])
        acc_z = np.array([d['az'] for d in data])
        acc_magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)

        # Extract features
        features = extract_gait_features(timestamps, acc_magnitude, acc_x, acc_y, acc_z)

        # Check for NaN features
        has_nan = any(np.isnan(v) for v in features.values())

        if has_nan or gait_model is None:
            # Fallback: use heuristic scoring
            cv = features.get('stride_time_cv', 5.0)
            if np.isnan(cv):
                cv = 5.0
            prob_pd = min(cv / 15.0, 1.0)  # Simple heuristic
            prediction = 1 if prob_pd > 0.5 else 0
            probability = [1 - prob_pd, prob_pd]
            shap_values = {name: 0.0 for name in gait_feature_names}
        else:
            # Scale and predict
            feature_vector = np.array([features.get(f, 0.0) for f in gait_feature_names]).reshape(1, -1)
            feature_vector = np.nan_to_num(feature_vector, nan=0.0)
            feature_scaled = gait_scaler.transform(feature_vector)

            prediction = int(gait_model.predict(feature_scaled)[0])
            probability = gait_model.predict_proba(feature_scaled)[0].tolist()

            shap_values = compute_shap_values(gait_model, feature_scaled, gait_feature_names)

        result = {
            'prediction': prediction,
            'probability_healthy': float(probability[0]),
            'probability_pd': float(probability[1]),
            'risk_level': 'Low' if probability[1] < 0.3 else 'Medium' if probability[1] < 0.7 else 'High',
            'features': {k: (float(v) if not np.isnan(v) else None) for k, v in features.items()},
            'shap_values': shap_values,
            'raw_signal': {
                'time': timestamps.tolist(),
                'magnitude': acc_magnitude.tolist()
            }
        }

        latest_results['gait'] = result
        update_combined_score()
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/gait/upload-csv', methods=['POST'])
def upload_csv():
    """Accept phyphox-exported CSV file (columns: time, acc_x, acc_y, acc_z)."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        content = file.read().decode('utf-8')
        reader = csv.reader(io.StringIO(content))

        rows = list(reader)
        # Find header or skip it
        start_row = 0
        for i, row in enumerate(rows):
            try:
                float(row[0])
                start_row = i
                break
            except (ValueError, IndexError):
                continue

        sensor_data = []
        for row in rows[start_row:]:
            try:
                t = float(row[0])
                ax = float(row[1])
                ay = float(row[2])
                az = float(row[3])
                sensor_data.append({'t': t, 'ax': ax, 'ay': ay, 'az': az})
            except (ValueError, IndexError):
                continue

        if len(sensor_data) < 50:
            return jsonify({'error': 'Not enough data in CSV'}), 400

        # Reuse the analyze endpoint logic
        request_data = {'sensor_data': sensor_data}
        # Call analyze_gait_internal
        return _analyze_gait_data(sensor_data)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


def _analyze_gait_data(data):
    """Internal function to process sensor data."""
    timestamps = np.array([d['t'] for d in data])
    acc_x = np.array([d['ax'] for d in data])
    acc_y = np.array([d['ay'] for d in data])
    acc_z = np.array([d['az'] for d in data])
    acc_magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)

    features = extract_gait_features(timestamps, acc_magnitude, acc_x, acc_y, acc_z)
    has_nan = any(np.isnan(v) for v in features.values())

    if has_nan or gait_model is None:
        cv = features.get('stride_time_cv', 5.0)
        if np.isnan(cv):
            cv = 5.0
        prob_pd = min(cv / 15.0, 1.0)
        prediction = 1 if prob_pd > 0.5 else 0
        probability = [1 - prob_pd, prob_pd]
        shap_values = {name: 0.0 for name in gait_feature_names}
    else:
        feature_vector = np.array([features.get(f, 0.0) for f in gait_feature_names]).reshape(1, -1)
        feature_vector = np.nan_to_num(feature_vector, nan=0.0)
        feature_scaled = gait_scaler.transform(feature_vector)
        prediction = int(gait_model.predict(feature_scaled)[0])
        probability = gait_model.predict_proba(feature_scaled)[0].tolist()
        shap_values = compute_shap_values(gait_model, feature_scaled, gait_feature_names)

    result = {
        'prediction': prediction,
        'probability_healthy': float(probability[0]),
        'probability_pd': float(probability[1]),
        'risk_level': 'Low' if probability[1] < 0.3 else 'Medium' if probability[1] < 0.7 else 'High',
        'features': {k: (float(v) if not np.isnan(v) else None) for k, v in features.items()},
        'shap_values': shap_values,
        'raw_signal': {
            'time': timestamps.tolist(),
            'magnitude': acc_magnitude.tolist()
        }
    }

    latest_results['gait'] = result
    update_combined_score()
    return jsonify(result)


@app.route('/api/tapping/analyze', methods=['POST'])
def analyze_tapping():
    """Receive tap timestamps from phone."""
    try:
        tap_data = request.json.get('tap_data', [])

        if len(tap_data) < 6:
            return jsonify({'error': 'Not enough taps recorded'}), 400

        features = extract_tapping_features(tap_data)
        if features is None:
            return jsonify({'error': 'Could not extract tapping features'}), 400

        risk_score, risk_level = assess_tapping_risk(features)

        # Compute intervals for visualization
        timestamps = np.array(tap_data)
        intervals = np.diff(timestamps) / 1000.0
        intervals = intervals[intervals < 2.0]

        result = {
            'risk_score': risk_score,
            'risk_level': risk_level,
            'features': features,
            'tap_intervals': intervals.tolist()
        }

        latest_results['tapping'] = result
        update_combined_score()
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/voice/analyze', methods=['POST'])
def analyze_voice():
    """
    Voice analysis. Accepts either:
    - multipart/form-data with 'audio' file (WAV/WebM) → extract features with parselmouth
    - application/json with 'features' dict → run directly through model
    - no data → demo mode
    """
    try:
        extracted_features = None
        features_dict = {}

        # --- Try real audio processing ---
        if 'audio' in request.files:
            audio_file = request.files['audio']
            import tempfile, os as _os
            suffix = '.webm' if audio_file.filename.endswith('.webm') else '.wav'
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                audio_file.save(tmp.name)
                tmp_path = tmp.name

            try:
                import parselmouth
                from parselmouth.praat import call
                import warnings
                warnings.filterwarnings('ignore')

                # Convert webm to wav if needed using subprocess
                wav_path = tmp_path
                if suffix == '.webm':
                    wav_path = tmp_path.replace('.webm', '.wav')
                    import subprocess
                    subprocess.run(['ffmpeg', '-i', tmp_path, wav_path, '-y', '-loglevel', 'quiet'], check=True)

                snd = parselmouth.Sound(wav_path)
                duration = snd.duration

                # Pitch (F0) analysis
                pitch = snd.to_pitch()
                pitch_values = pitch.selected_array['frequency']
                pitch_values = pitch_values[pitch_values > 0]

                # Jitter (local) - use Praat
                point_process = call(snd, "To PointProcess (periodic, cc)", 75, 500)
                jitter_local = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
                jitter_rap = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)

                # Shimmer (local)
                shimmer_local = call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
                shimmer_apq3 = call([snd, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

                # HNR
                harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
                hnr = call(harmonicity, "Get mean", 0, 0)

                # NHR (noise-to-harmonics)
                nhr = 1.0 / (10 ** (hnr / 10)) if hnr > 0 else 1.0

                features_dict = {
                    'MDVP:Jitter(%)': float(jitter_local * 100) if jitter_local else 0.01,
                    'MDVP:Jitter(Abs)': float(jitter_local / pitch_values.mean()) if len(pitch_values) > 0 and jitter_local else 0.0,
                    'MDVP:RAP': float(jitter_rap) if jitter_rap else 0.01,
                    'MDVP:PPQ': float(jitter_rap) if jitter_rap else 0.01,
                    'Jitter:DDP': float(jitter_rap * 3) if jitter_rap else 0.01,
                    'MDVP:Shimmer': float(shimmer_local) if shimmer_local else 0.05,
                    'MDVP:Shimmer(dB)': float(-20 * np.log10(1 - shimmer_local)) if shimmer_local and shimmer_local < 1 else 0.5,
                    'Shimmer:APQ3': float(shimmer_apq3) if shimmer_apq3 else 0.03,
                    'Shimmer:APQ5': float(shimmer_apq3 * 1.2) if shimmer_apq3 else 0.04,
                    'MDVP:APQ': float(shimmer_apq3 * 1.5) if shimmer_apq3 else 0.05,
                    'Shimmer:DDA': float(shimmer_apq3 * 3) if shimmer_apq3 else 0.09,
                    'NHR': float(nhr),
                    'HNR': float(hnr),
                    'RPDE': 0.5,   # placeholder (needs nolds library)
                    'DFA': 0.7,    # placeholder
                    'spread1': float(np.std(pitch_values)) if len(pitch_values) > 0 else 0.5,
                    'spread2': float(np.var(pitch_values)) if len(pitch_values) > 0 else 0.5,
                    'D2': 2.0,     # placeholder
                    'PPE': 0.2,    # placeholder
                    'status': -1,  # unknown
                }
                extracted_features = features_dict
                _os.unlink(tmp_path)
                if wav_path != tmp_path and _os.path.exists(wav_path):
                    _os.unlink(wav_path)

            except Exception as e:
                print(f"Parselmouth feature extraction failed: {e}")
                _os.unlink(tmp_path)
                extracted_features = None

        # --- JSON features ---
        elif request.json and 'features' in request.json:
            extracted_features = request.json['features']

        # --- Run model ---
        if extracted_features and voice_model is not None:
            feature_vector = np.array([extracted_features.get(f, 0.0) for f in voice_feature_names]).reshape(1, -1)
            feature_vector = np.nan_to_num(feature_vector, nan=0.0)
            feature_scaled = voice_scaler.transform(feature_vector)
            prediction = int(voice_model.predict(feature_scaled)[0])
            probability = voice_model.predict_proba(feature_scaled)[0].tolist()
            features_dict = extracted_features
        else:
            # Demo mode — realistic result based on jitter/shimmer if available
            if extracted_features:
                jitter = extracted_features.get('MDVP:Jitter(%)', 0.5)
                prob_pd = min(float(jitter) / 1.0, 0.95)
            else:
                prob_pd = 0.72  # Demo: high risk example
            prediction = 1 if prob_pd > 0.5 else 0
            probability = [1 - prob_pd, prob_pd]
            features_dict = {'demo_mode': True}

        result = {
            'prediction': prediction,
            'probability_pd': float(probability[1]),
            'probability_healthy': float(probability[0]),
            'risk_level': 'Low' if probability[1] < 0.35 else 'Medium' if probability[1] < 0.65 else 'High',
            'features': {k: v for k, v in features_dict.items() if k != 'status'},
        }

        latest_results['voice'] = result
        update_combined_score()
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/server-info', methods=['GET'])
def server_info():
    """Return server's local IP (or ngrok URL) so QR code can use it."""
    # If ngrok tunnel is active, use that URL (works on iPhone over HTTPS)
    ngrok_url = os.environ.get('PARKINSIGHT_BASE_URL')
    if ngrok_url:
        host = ngrok_url.replace('https://', '').replace('http://', '').rstrip('/')
        return jsonify({'ip': host, 'port': '', 'base_url': ngrok_url, 'https': True})

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = '127.0.0.1'
    return jsonify({'ip': local_ip, 'port': 5000})


@app.route('/api/results/latest', methods=['GET'])
def get_latest_results():
    """Dashboard polls this endpoint."""
    return jsonify(latest_results)


@app.route('/api/results/reset', methods=['POST'])
def reset_results():
    """Clear all results for new patient/demo."""
    for key in latest_results:
        latest_results[key] = None
    return jsonify({'status': 'reset'})


@app.route('/api/demo/<scenario>', methods=['POST'])
def run_demo(scenario):
    """Load and process pre-recorded demo data."""
    try:
        demo = load_demo(scenario)
        if demo is None:
            return jsonify({'error': f'Demo scenario "{scenario}" not found'}), 404

        # Reset first
        for key in latest_results:
            latest_results[key] = None

        # Process gait demo
        if 'gait' in demo:
            gait_data = demo['gait']['sensor_data']
            timestamps = np.array([d['t'] for d in gait_data])
            acc_x = np.array([d['ax'] for d in gait_data])
            acc_y = np.array([d['ay'] for d in gait_data])
            acc_z = np.array([d['az'] for d in gait_data])
            acc_magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)

            features = extract_gait_features(timestamps, acc_magnitude, acc_x, acc_y, acc_z)
            has_nan = any(np.isnan(v) for v in features.values())

            if has_nan or gait_model is None:
                cv = features.get('stride_time_cv', 5.0)
                if np.isnan(cv):
                    cv = 5.0
                prob_pd = min(cv / 15.0, 1.0)
                prediction = 1 if prob_pd > 0.5 else 0
                probability = [1 - prob_pd, prob_pd]
                shap_vals = {name: 0.0 for name in gait_feature_names}
            else:
                feature_vector = np.array([features.get(f, 0.0) for f in gait_feature_names]).reshape(1, -1)
                feature_vector = np.nan_to_num(feature_vector, nan=0.0)
                feature_scaled = gait_scaler.transform(feature_vector)
                prediction = int(gait_model.predict(feature_scaled)[0])
                probability = gait_model.predict_proba(feature_scaled)[0].tolist()
                shap_vals = compute_shap_values(gait_model, feature_scaled, gait_feature_names)

            latest_results['gait'] = {
                'prediction': prediction,
                'probability_healthy': float(probability[0]),
                'probability_pd': float(probability[1]),
                'risk_level': 'Low' if probability[1] < 0.3 else 'Medium' if probability[1] < 0.7 else 'High',
                'features': {k: (float(v) if not np.isnan(v) else None) for k, v in features.items()},
                'shap_values': shap_vals,
                'raw_signal': {
                    'time': timestamps.tolist(),
                    'magnitude': acc_magnitude.tolist()
                }
            }

        # Process tapping demo
        if 'tapping' in demo:
            tap_data = demo['tapping']['tap_data']
            features = extract_tapping_features(tap_data)
            if features:
                risk_score, risk_level = assess_tapping_risk(features)
                timestamps = np.array(tap_data)
                intervals = np.diff(timestamps) / 1000.0
                intervals = intervals[intervals < 2.0]

                latest_results['tapping'] = {
                    'risk_score': risk_score,
                    'risk_level': risk_level,
                    'features': features,
                    'tap_intervals': intervals.tolist()
                }

        update_combined_score()
        return jsonify({'status': 'ok', 'scenario': scenario, 'results': latest_results})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


def update_combined_score():
    """Weighted combination of available test results."""
    scores = []
    weights = []

    if latest_results['gait']:
        scores.append(latest_results['gait']['probability_pd'])
        weights.append(0.5)
    if latest_results['voice']:
        scores.append(latest_results['voice']['probability_pd'])
        weights.append(0.3)
    if latest_results['tapping']:
        scores.append(latest_results['tapping']['risk_score'])
        weights.append(0.2)

    if scores:
        combined = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
        latest_results['combined'] = {
            'score': combined,
            'risk_level': 'Low' if combined < 0.3 else 'Medium' if combined < 0.7 else 'High',
            'tests_completed': len(scores)
        }


# ---- Serve Static Files ----

@app.route('/cert')
def download_cert():
    """Lets iPhone download and install the SSL certificate."""
    cert_path = os.path.join(os.path.dirname(__file__), 'certs', 'cert.pem')
    if not os.path.exists(cert_path):
        return 'No certificate found. Run: python backend/app.py --https first.', 404
    return send_from_directory(
        os.path.join(os.path.dirname(__file__), 'certs'),
        'cert.pem',
        mimetype='application/x-pem-file',
        as_attachment=True,
        download_name='parkinsight.pem'
    )


@app.route('/phone')
@app.route('/phone/')
def serve_phone():
    return send_from_directory(os.path.join(os.path.dirname(__file__), '..', 'phone-capture'), 'index.html')


@app.route('/dashboard')
@app.route('/dashboard/')
def serve_dashboard():
    return send_from_directory(os.path.join(os.path.dirname(__file__), '..', 'dashboard'), 'index.html')


@app.route('/dashboard/<path:path>')
def serve_dashboard_files(path):
    return send_from_directory(os.path.join(os.path.dirname(__file__), '..', 'dashboard'), path)


@app.route('/')
def index():
    """Root route — redirect to dashboard."""
    return serve_dashboard()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--https', action='store_true', help='Run with HTTPS (needed for iOS DeviceMotion)')
    args = parser.parse_args()

    # Get local IP
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = 'localhost'

    protocol = 'https' if args.https else 'http'
    print("\n" + "=" * 55)
    print("  ParkInsight Server")
    print("=" * 55)
    print(f"  Dashboard : {protocol}://{local_ip}:5000/dashboard")
    print(f"  Phone URL : {protocol}://{local_ip}:5000/phone")
    if args.https:
        print("\n  iOS note: Accept the browser security warning once,")
        print("  then the phone page will load normally.")
    else:
        print("\n  TIP — for iOS gait test, restart with:")
        print("    python backend/app.py --https")
    print("=" * 55 + "\n")

    if args.https:
        # Generate proper SAN cert (required by Safari/iOS)
        from gen_cert import generate, CERT_FILE, KEY_FILE
        generate(local_ip)
        print(f"\n  iPhone setup (one-time only):")
        print(f"  1. Open https://{local_ip}:5000/cert on iPhone Safari")
        print(f"     -> Tap 'Allow' to download the profile")
        print(f"  2. Settings > General > VPN & Device Management")
        print(f"     -> Tap ParkInsight Dev -> Install")
        print(f"  3. Settings > General > About > Certificate Trust Settings")
        print(f"     -> Enable full trust for ParkInsight Dev")
        print(f"  4. Open https://{local_ip}:5000/phone  — gait test will work!\n")
        app.run(host='0.0.0.0', port=5000, debug=False, ssl_context=(CERT_FILE, KEY_FILE))
    else:
        app.run(host='0.0.0.0', port=5000, debug=False)
