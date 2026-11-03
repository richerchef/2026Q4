import json
import os
import argparse
from datetime import datetime
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Set high visual quality styling
plt.style.use('dark_background')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

def parse_time_series(json_filepath):
    """
    Parses JSON time series input file.
    Expected formats:
    - [{ "timestamp": "2025-01-01T00:00:00Z", "value": 12.3 }, ...]
    - { "timestamps": ["2025-01-01T00:00:00Z", ...], "values": [12.3, ...] }
    """
    with open(json_filepath, 'r') as f:
        data = json.load(f)

    timestamps = []
    values = []

    if isinstance(data, list):
        for entry in data:
            timestamps.append(entry['timestamp'])
            values.append(float(entry['value']))
    elif isinstance(data, dict):
        timestamps = data.get('timestamps', [])
        values = data.get('values', [])
    else:
        raise ValueError("Unsupported JSON layout. Expected list of objects or dict of arrays.")

    # Parse ISO 8601 timestamps
    parsed_dates = []
    for ts in timestamps:
        # Handle trailing Z for UTC
        ts_clean = ts.replace('Z', '+00:00')
        parsed_dates.append(datetime.fromisoformat(ts_clean))

    # Convert dates to seconds relative to start
    t0 = parsed_dates[0].timestamp()
    time_sec = np.array([d.timestamp() - t0 for d in parsed_dates])
    values = np.array(values, dtype=np.float64)

    # Sort chronological order
    sort_idx = np.argsort(time_sec)
    return parsed_dates, time_sec[sort_idx], values[sort_idx]


def format_duration(seconds):
    """Formats a duration in seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        return f"{seconds / 60:.2f} minutes"
    elif seconds < 86400:
        return f"{seconds / 3600:.2f} hours"
    elif seconds < 31536000:
        return f"{seconds / 86400:.2f} days"
    else:
        return f"{seconds / 31536000:.2f} years"


def detect_periodicities(time_sec, values, max_candidates=3):
    """
    Uses normalized autocorrelation to find dominant cyclic periods.
    Returns sorted array of detected periods in seconds and scores.
    """
    total_duration = time_sec[-1] - time_sec[0]
    num_samples = min(8192, max(1024, len(time_sec)))
    sample_step_sec = total_duration / num_samples

    # Uniform resampling via linear interpolation
    resampled_times = np.linspace(time_sec[0], time_sec[-1], num_samples)
    resampled_vals = np.interp(resampled_times, time_sec, values)

    # Standardize signal
    norm_vals = resampled_vals - np.mean(resampled_vals)
    variance = np.var(norm_vals)
    if variance == 0:
        return []

    # Compute Autocorrelation
    autocorr = signal.fftconvolve(norm_vals, norm_vals[::-1], mode='full')
    autocorr = autocorr[len(norm_vals) - 1:] / (variance * len(norm_vals))

    max_lag = int(num_samples * 0.75)
    autocorr = autocorr[:max_lag]

    # Find initial zero-crossing / decay lobe
    start_lag = 1
    while start_lag < max_lag - 1 and autocorr[start_lag] > 0 and autocorr[start_lag] >= autocorr[start_lag + 1]:
        start_lag += 1

    # Peak detection in autocorrelation
    peaks, properties = signal.find_peaks(autocorr[start_lag:], height=0.08, distance=5)
    peaks = peaks + start_lag

    candidates = []
    for p in peaks:
        # Refine peak precision using parabolic interpolation
        if 0 < p < len(autocorr) - 1:
            y0, y1, y2 = autocorr[p - 1], autocorr[p], autocorr[p + 1]
            denom = (2 * y1 - y0 - y2)
            delta = (y2 - y0) / (2 * denom) if denom != 0 else 0
            refined_p = p + delta
            score = y1 + 0.25 * (y2 - y0) * delta
        else:
            refined_p = p
            score = autocorr[p]

        period_sec = refined_p * sample_step_sec
        candidates.append({'period_sec': period_sec, 'score': float(score)})

    # Sort candidates by smallest period first if score is comparable, to catch fundamental frequency over higher multiples
    candidates.sort(key=lambda x: x['period_sec'])

    # Filter out higher-order integer multiples (harmonics) so fundamental frequency is prioritized
    filtered = []
    for c in candidates:
        is_multiple = False
        for existing in filtered:
            ratio = c['period_sec'] / existing['period_sec']
            if abs(ratio - round(ratio)) < 0.08:
                is_multiple = True
                break
        if not is_multiple:
            filtered.append(c)

    # Re-sort remaining fundamental candidates by autocorrelation score
    filtered.sort(key=lambda x: x['score'], reverse=True)

    return filtered[:max_candidates]


def detect_event_bursts(time_sec, values, threshold_std=2.0):
    """
    Detects spike/burst events (such as valve openings or pump start-ups).
    Returns timestamps and intervals between consecutive spikes.
    """
    mean_val = np.mean(values)
    std_val = np.std(values)
    threshold = mean_val + threshold_std * std_val

    # Find local maxima above threshold
    peaks, _ = signal.find_peaks(values, height=threshold, distance=3)

    if len(peaks) < 2:
        return [], None, 0.0

    event_times = time_sec[peaks]
    intervals = np.diff(event_times)
    median_interval = float(np.median(intervals))

    return event_times, intervals, median_interval


def get_spoke_labels(period_sec):
    """Generates dynamic time labels for polar spokes based on period scale."""
    if period_sec <= 120:
        num_divs = 6
        step = period_sec / num_divs
        return [f"{i * step:.1f}s" for i in range(num_divs)]
    elif period_sec <= 7200:
        num_divs = 8 if (period_sec / 60) % 8 == 0 else 6
        step_min = (period_sec / 60) / num_divs
        labels = []
        for i in range(num_divs):
            m = i * step_min
            labels.append(f"{m:.1f}m" if m % 1 != 0 else f"{int(m)}m")
        return labels
    elif period_sec <= 259200:
        total_hours = period_sec / 3600.0
        num_divs = 12 if total_hours == 24 else 6
        step_hr = total_hours / num_divs
        if total_hours == 24:
            return [f"{int(i * step_hr):02d}:00" for i in range(num_divs)]
        return [f"+{i * step_hr:.1f}h" for i in range(num_divs)]
    else:
        total_days = period_sec / 86400.0
        if abs(total_days - 7) < 0.2:
            return ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        num_divs = 6
        step_d = total_days / num_divs
        return [f"Day {i * step_d:.1f}" for i in range(num_divs)]


def generate_spiral_chart(time_sec, values, period_sec, output_img_path, title_prefix="System Data"):
    """
    Renders polar spiral chart along with time-domain verification and time-based spokes.
    """
    fig = plt.figure(figsize=(14, 7), facecolor='#090d16')

    # --- LEFT PANEL: Spiral Plot ---
    ax_polar = fig.add_subplot(121, projection='polar', facecolor='#0b1120')

    # Map time into spiral coordinates
    turns = time_sec / period_sec
    theta = (turns * 2 * np.pi) - (np.pi / 2)  # Start top (12 o'clock)
    
    # Radius expands outwards as time progresses
    inner_r = 1.0
    r = inner_r + turns

    # Normalize values for color mapping
    norm_vals = (values - np.min(values)) / (np.ptp(values) if np.ptp(values) > 0 else 1.0)
    colors = cm.plasma(norm_vals)

    # Plot Scatter Spiral
    scatter = ax_polar.scatter(theta, r, c=colors, s=18, alpha=0.85, edgecolors='none')

    # Configure Time-Aware Spokes
    spoke_labels = get_spoke_labels(period_sec)
    num_spokes = len(spoke_labels)
    spoke_angles = [(i / num_spokes) * 360 for i in range(num_spokes)]

    ax_polar.set_theta_zero_location('N')
    ax_polar.set_theta_direction(-1)  # Clockwise rotation
    ax_polar.set_thetagrids(spoke_angles, labels=spoke_labels, color='#94a3b8', fontsize=9)
    ax_polar.set_rticks([])  # Hide radial distance numbers for clean look
    ax_polar.grid(True, color='#334155', linestyle='--', alpha=0.5)

    period_str = format_duration(period_sec)
    ax_polar.set_title(f"ChronoSpiral View\nAligned to Period: {period_str}", color='#f8fafc', fontsize=12, pad=15)

    # --- RIGHT PANEL: Time Domain Signal ---
    ax_line = fig.add_subplot(122, facecolor='#0b1120')
    time_hours = time_sec / 3600.0
    ax_line.plot(time_hours, values, color='#38bdf8', linewidth=0.8, alpha=0.85)
    ax_line.set_xlabel('Elapsed Time (Hours)', color='#94a3b8')
    ax_line.set_ylabel('Sensor Value', color='#94a3b8')
    ax_line.set_title("Time Domain Sensor Readings", color='#f8fafc', fontsize=12)
    ax_line.grid(True, color='#1e293b', linestyle=':')
    ax_line.tick_params(colors='#94a3b8')

    plt.tight_layout()
    plt.savefig(output_img_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()


def generate_report(time_sec, values, candidates, event_times, intervals, median_interval, primary_period, img_filename):
    """Generates structured diagnosis text report."""
    total_time_str = format_duration(time_sec[-1] - time_sec[0])
    
    report = []
    report.append("==========================================================")
    report.append("      AUTOMATED TIME SERIES SEASONALITY & DIAGNOSTIC REPORT")
    report.append("==========================================================")
    report.append(f"Analyzed Duration : {total_time_str}")
    report.append(f"Total Sample Points: {len(values):,}")
    report.append(f"Value Range        : Min = {np.min(values):.2f}, Max = {np.max(values):.2f}, Mean = {np.mean(values):.2f}")
    report.append("----------------------------------------------------------\n")

    report.append("1. DOMINANT PERIODICITY CANDIDATES (FFT / Autocorrelation):")
    if candidates:
        for idx, cand in enumerate(candidates, 1):
            dur_str = format_duration(cand['period_sec'])
            confidence = cand['score'] * 100
            report.append(f"   [{idx}] Fundamental Period: {dur_str:<18} (Autocorrelation Score: {confidence:.1f}%)")
    else:
        report.append("   No recurring cyclic periodicity detected in signal.")

    report.append("\n2. EVENT & ANOMALY BURST DETECTION:")
    if event_times is not None and len(event_times) >= 2:
        num_spikes = len(event_times)
        freq_str = format_duration(median_interval)
        report.append(f"   - Detected {num_spikes} prominent spike events / bursts.")
        report.append(f"   - Event Recurrence Interval: Occurs every {freq_str}.")

        # Check if multiple bursts occur per selected period
        if primary_period > median_interval * 1.5:
            occurrences_per_period = round(primary_period / median_interval)
            report.append(f"   - OCCURRENCE FREQUENCY: Detected ~{occurrences_per_period} instances per {format_duration(primary_period)} period cycle.")

        # Operational Insights
        if median_interval < 900:  # < 15 mins
            report.append(f"   - DIAGNOSTIC ALERT: High-frequency cycles detected (every {freq_str}).")
            report.append("     Possible cause: Relief valve rapid cycling, chatter, or small system pressure leak.")
        elif median_interval < 14400:  # < 4 hours
            report.append(f"   - DIAGNOSTIC ALERT: Regular mid-frequency refills/resets (every {freq_str}).")
            report.append("     Possible cause: Automatic tank topping-up, periodic sump pump activation, or scheduled purging.")
        else:
            report.append(f"   - DIAGNOSTIC ALERT: Slow-cycle process observed (every {freq_str}).")
    else:
        report.append("   - No sharp periodic spike events detected above standard threshold.")

    report.append("\n3. VISUALIZATION REPORT:")
    report.append(f"   - Generated spiral plot saved to: {img_filename}")
    report.append("==========================================================")

    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="ChronoSpiral Time Series Seasonality & Leak Detector")
    parser.add_argument("--input", type=str, default="sample_data.json", help="Path to input JSON time series")
    parser.add_argument("--output_img", type=str, default="seasonal_report.png", help="Path for output chart image")
    args = parser.parse_args()

    # Generate sample data if input file doesn't exist
    if not os.path.exists(args.input):
        print(f"Input file '{args.input}' not found. Generating sample dataset with 11.2 min relief valve cycle...")
        sample_data = []
        t0 = datetime(2025, 1, 1, 0, 0, 0)
        # 24 hours of 15-second data
        for i in range(5760):
            t_sec = i * 15
            dt = datetime.fromtimestamp(t0.timestamp() + t_sec)
            # 11.2 min valve open cycle (672 seconds)
            valve_spike = 40.0 if (t_sec % 672) < 30 else 0.0
            val = 10.0 + valve_spike + np.random.normal(0, 0.8)
            sample_data.append({"timestamp": dt.isoformat() + "Z", "value": val})
        
        with open(args.input, "w") as f:
            json.dump(sample_data, f)

    # 1. Parse Data
    dates, time_sec, values = parse_time_series(args.input)

    # 2. Detect Periodicities
    candidates = detect_periodicities(time_sec, values)

    # 3. Detect Burst Events
    event_times, intervals, median_interval = detect_event_bursts(time_sec, values)

    # Select fundamental primary period for spiral plot
    if candidates:
        primary_period = candidates[0]['period_sec']
    elif median_interval > 0:
        primary_period = median_interval
    else:
        primary_period = 86400

    # 4. Render Spiral Chart
    generate_spiral_chart(time_sec, values, primary_period, args.output_img)

    # 5. Output Report
    report_text = generate_report(time_sec, values, candidates, event_times, intervals, median_interval, primary_period, args.output_img)
    print("\n" + report_text)


if __name__ == "__main__":
    main()
