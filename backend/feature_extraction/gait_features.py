import numpy as np
from scipy.signal import find_peaks, butter, filtfilt
from scipy.stats import iqr


def get_feature_names():
    """Return ordered list of feature names (must match model training order)."""
    return [
        'stride_time_mean', 'stride_time_std', 'stride_time_cv',
        'stride_time_iqr', 'stride_time_range', 'stride_time_mad',
        'cadence', 'stride_asymmetry', 'cv_asymmetry',
        'mean_peak_acc', 'peak_acc_variability',
        'step_regularity', 'stride_regularity',
        'dfa_alpha', 'sample_entropy', 'swing_stance_ratio',
        'stride_time_skew', 'stride_time_kurtosis'
    ]


def _dfa(data, min_box=4, max_box=None, num_scales=20):
    """Detrended Fluctuation Analysis — returns scaling exponent alpha."""
    N = len(data)
    if N < 16:
        return 0.5

    if max_box is None:
        max_box = N // 4

    # Integrate the signal (cumulative sum of deviations from mean)
    y = np.cumsum(data - np.mean(data))

    scales = np.unique(np.logspace(
        np.log10(min_box), np.log10(max_box), num_scales
    ).astype(int))
    scales = scales[scales >= 4]

    fluctuations = []
    valid_scales = []

    for scale in scales:
        n_segments = N // scale
        if n_segments < 1:
            continue

        rms_values = []
        for i in range(n_segments):
            segment = y[i * scale:(i + 1) * scale]
            x_fit = np.arange(len(segment))
            coeffs = np.polyfit(x_fit, segment, 1)
            trend = np.polyval(coeffs, x_fit)
            rms_values.append(np.sqrt(np.mean((segment - trend) ** 2)))

        if rms_values:
            fluctuations.append(np.mean(rms_values))
            valid_scales.append(scale)

    if len(valid_scales) < 3:
        return 0.5

    log_scales = np.log(valid_scales)
    log_fluct = np.log(fluctuations)

    # Linear fit in log-log space
    coeffs = np.polyfit(log_scales, log_fluct, 1)
    return float(coeffs[0])


def _sample_entropy(data, m=2, r_factor=0.2):
    """Sample entropy of a time series."""
    N = len(data)
    if N < 10:
        return 0.0

    r = r_factor * np.std(data)
    if r == 0:
        return 0.0

    def _count_matches(template_len):
        count = 0
        templates = []
        for i in range(N - template_len):
            templates.append(data[i:i + template_len])
        templates = np.array(templates)

        for i in range(len(templates)):
            # Chebyshev distance
            dists = np.max(np.abs(templates[i] - templates), axis=1)
            count += np.sum(dists < r) - 1  # Exclude self-match

        return count

    A = _count_matches(m + 1)
    B = _count_matches(m)

    if B == 0 or A == 0:
        return 0.0

    return float(-np.log(A / B))


def extract_gait_features_from_vgrf(timestamps, total_force_left, total_force_right):
    """
    Extract gait features from PhysioNet VGRF data (foot force sensors).
    """
    dt = np.median(np.diff(timestamps))
    fs = 1.0 / dt

    features = {}
    stride_data = {}

    for side, signal in [('left', total_force_left), ('right', total_force_right)]:
        # Low-pass filter
        nyquist = fs / 2
        cutoff = min(15.0, nyquist * 0.9)
        b, a = butter(4, cutoff / nyquist, btype='low')
        filtered = filtfilt(b, a, signal)

        # Detect heel strikes — use adaptive threshold
        min_distance = int(fs * 0.4)  # Min 0.4s between same-foot strikes
        height_threshold = np.max(filtered) * 0.15

        peaks, props = find_peaks(
            filtered,
            height=height_threshold,
            distance=min_distance,
            prominence=np.max(filtered) * 0.05
        )

        if len(peaks) < 4:
            return {name: float('nan') for name in get_feature_names()}

        # Stride intervals (consecutive peaks of same foot)
        stride_times = np.diff(timestamps[peaks])
        # Remove outliers
        stride_times = stride_times[(stride_times > 0.4) & (stride_times < 3.0)]

        if len(stride_times) < 3:
            return {name: float('nan') for name in get_feature_names()}

        stride_data[side] = {
            'stride_times': stride_times,
            'peaks': peaks,
            'filtered': filtered
        }

    stride_l = stride_data['left']['stride_times']
    stride_r = stride_data['right']['stride_times']
    peaks_l = stride_data['left']['peaks']
    peaks_r = stride_data['right']['peaks']

    # Temporal features (average of both feet)
    all_strides = np.concatenate([stride_l, stride_r])
    features['stride_time_mean'] = float(np.mean(all_strides))
    features['stride_time_std'] = float(np.std(all_strides))
    features['stride_time_cv'] = float(np.std(all_strides) / np.mean(all_strides) * 100)

    # Variability features
    features['stride_time_iqr'] = float(iqr(all_strides))
    features['stride_time_range'] = float(np.ptp(all_strides))
    features['stride_time_mad'] = float(np.median(np.abs(all_strides - np.median(all_strides))))

    # Cadence (steps per minute) — each peak is a stride, stride = 2 steps
    features['cadence'] = float(120.0 / np.mean(all_strides)) if np.mean(all_strides) > 0 else 0

    # Asymmetry
    mean_l = np.mean(stride_l)
    mean_r = np.mean(stride_r)
    features['stride_asymmetry'] = float(abs(mean_l - mean_r) / (0.5 * (mean_l + mean_r)))

    cv_l = np.std(stride_l) / np.mean(stride_l) * 100
    cv_r = np.std(stride_r) / np.mean(stride_r) * 100
    features['cv_asymmetry'] = float(abs(cv_l - cv_r))

    # Force features
    combined_signal = total_force_left + total_force_right
    nyquist = fs / 2
    cutoff = min(15.0, nyquist * 0.9)
    b, a = butter(4, cutoff / nyquist, btype='low')
    combined_filtered = filtfilt(b, a, combined_signal)

    all_peaks = np.sort(np.concatenate([peaks_l, peaks_r]))
    peak_values = combined_filtered[all_peaks]
    features['mean_peak_acc'] = float(np.mean(peak_values))
    features['peak_acc_variability'] = float(np.std(peak_values) / np.mean(peak_values) * 100) if np.mean(peak_values) > 0 else 0

    # Step regularity (autocorrelation)
    acc_centered = combined_filtered - np.mean(combined_filtered)
    n_auto = min(len(acc_centered), int(5 * fs))  # Limit autocorr computation
    acc_short = acc_centered[:n_auto]
    autocorr = np.correlate(acc_short, acc_short, mode='full')
    autocorr = autocorr[len(autocorr) // 2:]
    if autocorr[0] != 0:
        autocorr = autocorr / autocorr[0]

    step_intervals_all = []
    for i in range(1, len(all_peaks)):
        si = timestamps[all_peaks[i]] - timestamps[all_peaks[i - 1]]
        if 0.2 < si < 1.5:
            step_intervals_all.append(si)

    if len(step_intervals_all) > 0:
        step_lag = int(np.mean(step_intervals_all) * fs)
        stride_lag = step_lag * 2

        search_start = max(1, step_lag - int(0.15 * fs))
        search_end = min(len(autocorr), step_lag + int(0.15 * fs))
        features['step_regularity'] = float(np.max(autocorr[search_start:search_end])) if search_end > search_start else 0.0

        search_start2 = max(1, stride_lag - int(0.25 * fs))
        search_end2 = min(len(autocorr), stride_lag + int(0.25 * fs))
        features['stride_regularity'] = float(np.max(autocorr[search_start2:search_end2])) if search_end2 > search_start2 else 0.0
    else:
        features['step_regularity'] = 0.0
        features['stride_regularity'] = 0.0

    # DFA (Detrended Fluctuation Analysis) — key PD biomarker
    # Use the longer stride series for DFA
    longer_strides = stride_l if len(stride_l) >= len(stride_r) else stride_r
    features['dfa_alpha'] = _dfa(longer_strides)

    # Sample Entropy — measures complexity/regularity of stride time series
    features['sample_entropy'] = _sample_entropy(longer_strides)

    # Swing-stance ratio approximation using force data
    # Swing phase = foot off ground (force near zero)
    left_swing_frac = float(np.mean(total_force_left < np.max(total_force_left) * 0.05))
    right_swing_frac = float(np.mean(total_force_right < np.max(total_force_right) * 0.05))
    avg_swing = (left_swing_frac + right_swing_frac) / 2
    features['swing_stance_ratio'] = float(avg_swing / (1 - avg_swing)) if avg_swing < 1 else 1.0

    # Higher-order statistics of stride times
    from scipy.stats import skew, kurtosis
    features['stride_time_skew'] = float(skew(all_strides))
    features['stride_time_kurtosis'] = float(kurtosis(all_strides))

    return features


def extract_gait_features(timestamps, acc_magnitude, acc_x, acc_y, acc_z):
    """
    Extract gait features from raw phone accelerometer data.
    """
    dt = np.median(np.diff(timestamps))
    fs = 1.0 / dt

    # Low-pass filter
    nyquist = fs / 2
    cutoff = min(15.0, nyquist * 0.9)
    b, a = butter(4, cutoff / nyquist, btype='low')
    acc_filtered = filtfilt(b, a, acc_magnitude)

    # Detect steps
    min_distance = int(fs * 0.4)
    height_threshold = np.percentile(acc_filtered, 60)

    peaks, _ = find_peaks(
        acc_filtered,
        height=height_threshold,
        distance=min_distance,
        prominence=0.3
    )

    if len(peaks) < 6:
        return {name: float('nan') for name in get_feature_names()}

    step_times = timestamps[peaks]
    step_intervals = np.diff(step_times)

    # Stride intervals (every other peak — same foot)
    stride_intervals = np.diff(step_times[::2])
    stride_intervals_r = np.diff(step_times[1::2])

    # Remove outliers
    stride_intervals = stride_intervals[(stride_intervals > 0.5) & (stride_intervals < 2.5)]
    stride_intervals_r = stride_intervals_r[(stride_intervals_r > 0.5) & (stride_intervals_r < 2.5)]
    step_intervals = step_intervals[(step_intervals > 0.25) & (step_intervals < 1.5)]

    if len(stride_intervals) < 3 or len(stride_intervals_r) < 3:
        return {name: float('nan') for name in get_feature_names()}

    features = {}

    all_strides = np.concatenate([stride_intervals, stride_intervals_r])

    # Temporal features
    features['stride_time_mean'] = float(np.mean(all_strides))
    features['stride_time_std'] = float(np.std(all_strides))
    features['stride_time_cv'] = float(np.std(all_strides) / np.mean(all_strides) * 100)

    # Variability features
    features['stride_time_iqr'] = float(iqr(all_strides))
    features['stride_time_range'] = float(np.ptp(all_strides))
    features['stride_time_mad'] = float(np.median(np.abs(all_strides - np.median(all_strides))))

    # Cadence
    features['cadence'] = float(60.0 / np.mean(step_intervals)) if np.mean(step_intervals) > 0 else 0

    # Asymmetry
    mean_l = np.mean(stride_intervals)
    mean_r = np.mean(stride_intervals_r)
    features['stride_asymmetry'] = float(abs(mean_l - mean_r) / (0.5 * (mean_l + mean_r)))

    cv_l = np.std(stride_intervals) / np.mean(stride_intervals) * 100
    cv_r = np.std(stride_intervals_r) / np.mean(stride_intervals_r) * 100
    features['cv_asymmetry'] = float(abs(cv_l - cv_r))

    # Acceleration features
    peak_values = acc_filtered[peaks]
    features['mean_peak_acc'] = float(np.mean(peak_values))
    features['peak_acc_variability'] = float(np.std(peak_values) / np.mean(peak_values) * 100)

    # Step regularity (autocorrelation)
    acc_centered = acc_filtered - np.mean(acc_filtered)
    n_auto = min(len(acc_centered), int(5 * fs))
    acc_short = acc_centered[:n_auto]
    autocorr = np.correlate(acc_short, acc_short, mode='full')
    autocorr = autocorr[len(autocorr) // 2:]
    if autocorr[0] != 0:
        autocorr = autocorr / autocorr[0]

    step_lag = int(np.mean(step_intervals) * fs) if len(step_intervals) > 0 else int(0.5 * fs)
    stride_lag = step_lag * 2

    search_start = max(1, step_lag - int(0.15 * fs))
    search_end = min(len(autocorr), step_lag + int(0.15 * fs))
    features['step_regularity'] = float(np.max(autocorr[search_start:search_end])) if search_end > search_start else 0.0

    search_start2 = max(1, stride_lag - int(0.25 * fs))
    search_end2 = min(len(autocorr), stride_lag + int(0.25 * fs))
    features['stride_regularity'] = float(np.max(autocorr[search_start2:search_end2])) if search_end2 > search_start2 else 0.0

    # DFA
    longer_strides = stride_intervals if len(stride_intervals) >= len(stride_intervals_r) else stride_intervals_r
    features['dfa_alpha'] = _dfa(longer_strides)

    # Sample Entropy
    features['sample_entropy'] = _sample_entropy(longer_strides)

    # Swing-stance ratio (approximate from acceleration)
    below_mean = np.mean(acc_filtered < np.mean(acc_filtered))
    features['swing_stance_ratio'] = float(below_mean / (1 - below_mean)) if below_mean < 1 else 1.0

    # Higher-order statistics
    from scipy.stats import skew, kurtosis
    features['stride_time_skew'] = float(skew(all_strides))
    features['stride_time_kurtosis'] = float(kurtosis(all_strides))

    return features
