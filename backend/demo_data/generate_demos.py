"""Generate pre-recorded demo datasets for fallback during presentations."""
import json
import numpy as np
import os

OUTPUT_DIR = os.path.dirname(__file__)


def generate_walking_signal(duration=30, fs=60, stride_cv=2.0, cadence=115, asymmetry=0.02):
    """
    Generate synthetic accelerometer data mimicking walking gait.

    Args:
        duration: seconds of walking
        fs: sampling rate (Hz)
        stride_cv: coefficient of variation for stride time (%)
        cadence: steps per minute
        asymmetry: left-right asymmetry factor
    """
    t = np.arange(0, duration, 1.0/fs)
    step_period = 60.0 / cadence  # seconds per step

    # Generate step times with variability
    step_times = []
    current_time = 0.5
    step_count = 0
    while current_time < duration - 0.5:
        # Alternate left/right with slight asymmetry
        if step_count % 2 == 0:
            period = step_period * (1 + asymmetry)
        else:
            period = step_period * (1 - asymmetry)

        # Add variability
        period *= (1 + np.random.normal(0, stride_cv / 100))
        step_times.append(current_time)
        current_time += period
        step_count += 1

    # Generate acceleration signal
    gravity = 9.81
    acc_magnitude = np.ones_like(t) * gravity

    for st in step_times:
        # Each step creates a peak in acceleration
        idx = np.argmin(np.abs(t - st))
        # Gaussian bump for step impact
        window = int(0.1 * fs)
        start = max(0, idx - window)
        end = min(len(t), idx + window)
        for i in range(start, end):
            dt_step = t[i] - st
            acc_magnitude[i] += 3.0 * np.exp(-dt_step**2 / (2 * 0.02**2))

    # Add noise
    acc_magnitude += np.random.normal(0, 0.3, len(t))

    # Split into components (phone in pocket orientation)
    acc_x = np.random.normal(0, 0.5, len(t))
    acc_y = acc_magnitude * 0.85 + np.random.normal(0, 0.2, len(t))
    acc_z = np.random.normal(0, 0.5, len(t))

    sensor_data = []
    for i in range(len(t)):
        sensor_data.append({
            't': round(float(t[i]), 4),
            'ax': round(float(acc_x[i]), 3),
            'ay': round(float(acc_y[i]), 3),
            'az': round(float(acc_z[i]), 3)
        })

    return sensor_data


def generate_tapping_data(duration=10, mean_interval=0.25, cv=5, fatigue=1.0):
    """Generate synthetic tapping timestamps."""
    timestamps = []
    current = 1000  # Start at 1 second (ms)

    while current < duration * 1000:
        timestamps.append(int(current))
        interval = mean_interval * 1000  # ms
        interval *= (1 + np.random.normal(0, cv / 100))
        # Apply fatigue
        progress = (current - 1000) / (duration * 1000)
        interval *= (1 + (fatigue - 1) * progress)
        current += max(interval, 100)

    return timestamps


if __name__ == '__main__':
    np.random.seed(42)

    # Scenario 1: Healthy walk
    healthy_gait = generate_walking_signal(
        duration=30, fs=60, stride_cv=2.0, cadence=115, asymmetry=0.02
    )
    healthy_tapping = generate_tapping_data(
        duration=10, mean_interval=0.25, cv=5, fatigue=1.02
    )

    # Scenario 2: Early PD
    early_pd_gait = generate_walking_signal(
        duration=30, fs=60, stride_cv=5.0, cadence=105, asymmetry=0.08
    )
    early_pd_tapping = generate_tapping_data(
        duration=10, mean_interval=0.35, cv=12, fatigue=1.12
    )

    # Scenario 3: Advanced PD
    advanced_pd_gait = generate_walking_signal(
        duration=30, fs=60, stride_cv=9.0, cadence=90, asymmetry=0.15
    )
    advanced_pd_tapping = generate_tapping_data(
        duration=10, mean_interval=0.55, cv=20, fatigue=1.3
    )

    demos = {
        'healthy': {
            'gait': {'sensor_data': healthy_gait},
            'tapping': {'tap_data': healthy_tapping}
        },
        'early_pd': {
            'gait': {'sensor_data': early_pd_gait},
            'tapping': {'tap_data': early_pd_tapping}
        },
        'advanced_pd': {
            'gait': {'sensor_data': advanced_pd_gait},
            'tapping': {'tap_data': advanced_pd_tapping}
        }
    }

    for name, data in demos.items():
        filepath = os.path.join(OUTPUT_DIR, f'{name}.json')
        with open(filepath, 'w') as f:
            json.dump(data, f)
        print(f"Generated {filepath} ({len(data['gait']['sensor_data'])} gait samples, {len(data['tapping']['tap_data'])} taps)")

    print("\nDone! Demo data files generated.")
