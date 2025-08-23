import av
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from base import analyze_video_metadata

import ruptures as rpt
from utils import timeit_log

# Normalize metrics to [0, 1] range
def normalize(arr):
    arr = np.array(arr)
    return (arr - arr.min()) / (arr.max() - arr.min()) if arr.max() > arr.min() else arr


@timeit_log
def detect_change_points(arr, frame_times):
    """
    Interpolates arr and frame_times to at least 500 points, then runs ruptures change point detection.
    Returns breakpoints and interpolated frame_times.
    """
    min_points = 500
    n = len(frame_times)
    if n < min_points and n > 1:
        interp_times = np.linspace(frame_times[0], frame_times[-1], min_points)
        arr_interp = np.zeros((min_points, arr.shape[1]))
        for i in range(arr.shape[1]):
            arr_interp[:, i] = np.interp(interp_times, frame_times, arr[:, i])
        frame_times = interp_times
        arr = arr_interp
        n = min_points
    algo = rpt.BottomUp(model="l2").fit(arr)
    bkps = algo.predict(n_bkps=np.ceil(n ** 0.5))
    # sigma = arr.std()
    # penalty = np.log(n) * arr.shape[1] * sigma**2
    # bkps = algo.predict(pen=penalty)
    return bkps, frame_times

def plot_analysis(results, save_path="video_analysis.png"):
    """Clean visualization focused on interest score."""
    
    frame_times = np.array(results.get("i_frame_times", []))
    n = len(frame_times)
    print('Got results...', n, len(results.get("gop_bitrate")), 
          len(results.get("i_frame_size")), len(results.get("gop_variances")))

    # Normalize frame_times so first frame is at 0 seconds
    if n > 0:
        frame_times = frame_times - frame_times[0]

    gop_bitrate = normalize(results["gop_bitrate"])
    i_frame_size = normalize(results["i_frame_size"])
    gop_variances = normalize(results["gop_variances"])

    arr = np.vstack([gop_bitrate, i_frame_size, gop_variances]).T
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(frame_times, gop_bitrate, label="GOP Bitrate", color="#2E86C1", linewidth=2)
    ax.plot(frame_times, i_frame_size, label="I-frame Size", color="#E74C3C", linewidth=2)
    ax.plot(frame_times, gop_variances, color="#28B463", linewidth=2, label="GOP Variance")

    # Draw keyframes (I-frames) as yellow circles
    # ax.scatter(frame_times, [1.05]*n, color="#F1C40F", marker="o", s=80, label="Keyframe I")

    # Detect change points (ruptures)
    bkps, frame_times_interp = detect_change_points(arr, frame_times)
    rupture_times = []
    for bkp in bkps[:-1]:
        if bkp < len(frame_times_interp):
            rt = frame_times_interp[bkp]
            rupture_times.append(rt)
            ax.axvline(rt, color='k', linestyle='--', alpha=0.7, label='Change Point' if bkp == bkps[0] else None)
    from matplotlib.ticker import FuncFormatter
    import datetime
    def seconds_to_hms(x, pos):
        return str(datetime.timedelta(seconds=max(0, int(x))))
    ax.xaxis.set_major_formatter(FuncFormatter(seconds_to_hms))
    # Limit base ticks for readability
    base_ticks = ax.get_xticks()
    base_ticks = base_ticks[::max(1, len(base_ticks)//10)]  # at most 10 regular ticks
    all_ticks = sorted(set(list(base_ticks) + rupture_times))
    ax.set_xticks(all_ticks)
    ax.set_xticklabels([seconds_to_hms(x, None) for x in all_ticks], rotation=45, fontsize=9)
    ax.set_ylabel("Normalized Value")
    ax.set_xlabel("Frame Time (hh:mm:ss)")
    ax.set_title("Video Metrics & Ruptures")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    print(f"Saved analysis: {save_path}")
    return fig


if __name__ == "__main__":
    try:
        results = analyze_video_metadata("./data/long.mp4")
        plot_analysis(results)
    except FileNotFoundError:
        print("❌ Error: Video file not found")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()