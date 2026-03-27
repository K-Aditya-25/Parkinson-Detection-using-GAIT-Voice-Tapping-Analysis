import numpy as np


def extract_tapping_features(tap_timestamps_ms):
    """Extract tapping features from raw tap timestamps (in milliseconds)."""
    timestamps = np.array(tap_timestamps_ms, dtype=float)
    intervals = np.diff(timestamps) / 1000.0  # ms -> seconds

    # Remove pauses (> 2s gaps = user stopped)
    intervals = intervals[intervals < 2.0]

    if len(intervals) < 5:
        return None

    half = len(intervals) // 2

    features = {
        'total_taps': int(len(timestamps)),
        'mean_interval': float(np.mean(intervals)),
        'std_interval': float(np.std(intervals)),
        'cv_interval': float(np.std(intervals) / np.mean(intervals) * 100),
        'median_interval': float(np.median(intervals)),
        'fatigue_ratio': float(np.mean(intervals[half:]) / np.mean(intervals[:half])) if half > 0 else 1.0,
        'rhythm_regularity': float(max(0.0, 1.0 - np.std(intervals) / np.mean(intervals))),
        'taps_per_second': float(len(timestamps) / ((timestamps[-1] - timestamps[0]) / 1000.0)) if len(timestamps) > 1 else 0.0,
        'interval_trend': float(np.polyfit(np.arange(len(intervals)), intervals, 1)[0]) if len(intervals) > 2 else 0.0,
    }

    return features


def assess_tapping_risk(features):
    """
    Continuous sigmoid-based risk scoring (not step thresholds).
    Produces a smooth 0-1 score that varies with each person's data.

    Clinical reference values from published research:
    - Healthy: CV ~5-8%, fatigue ratio ~1.0, mean interval ~0.25-0.30s
    - PD:      CV ~15-25%, fatigue ratio ~1.15-1.3, mean interval ~0.4-0.6s
    """
    def sigmoid(x, center, steepness=1.0):
        """Maps x to 0-1, with 0.5 at center."""
        return 1.0 / (1.0 + np.exp(-steepness * (x - center)))

    # Component 1: CV interval (healthy <8%, PD >15%)
    cv = features['cv_interval']
    cv_score = sigmoid(cv, center=11.0, steepness=0.18)  # 0.5 at CV=11%

    # Component 2: Fatigue ratio (healthy ~1.0, PD >1.15)
    fr = features['fatigue_ratio']
    fr_score = sigmoid(fr - 1.0, center=0.12, steepness=20.0)  # 0.5 at 12% slowdown

    # Component 3: Speed - mean interval (healthy <0.30s, PD >0.45s)
    mi = features['mean_interval']
    speed_score = sigmoid(mi, center=0.37, steepness=10.0)  # 0.5 at 370ms

    # Component 4: Interval trend (positive = slowing down over time)
    trend = features.get('interval_trend', 0.0)
    trend_score = sigmoid(trend * 1000, center=5.0, steepness=0.15)  # trend in ms/tap

    # Weighted combination
    risk_score = (
        0.35 * cv_score +
        0.25 * fr_score +
        0.30 * speed_score +
        0.10 * trend_score
    )

    risk_score = float(np.clip(risk_score, 0.0, 1.0))
    risk_level = 'Low' if risk_score < 0.35 else 'Medium' if risk_score < 0.65 else 'High'

    return risk_score, risk_level
