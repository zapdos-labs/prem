import av
import numpy as np
import matplotlib.pyplot as plt
import ruptures as rpt
from base import analyze_video_metadata
from utils import timeit_log


# Vectorized normalization (in-place for speed)
def normalize_array(arr):
    """
    Normalize each column of arr to [0, 1].
    Handles zero-variance columns by leaving them as zeros.
    """
    arr = np.asarray(arr, dtype=float)
    min_vals = arr.min(axis=0)
    max_vals = arr.max(axis=0)
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1  # avoid division by zero
    return (arr - min_vals) / ranges


@timeit_log
def detect_change_points(arr, min_points=200,  use_penalty=True):
    """
    Interpolates arr to at least `min_points` points if n < min_points,
    runs ruptures change point detection with optimized settings.

    Returns:
        bkps_original_idx: breakpoints mapped back to original indices
        bkps_interp_idx: breakpoints in interpolated index space
    """
    n = arr.shape[0]
    original_idx = np.arange(n)

    # Interpolation only if necessary
    if n < min_points and n > 1:
        interp_idx = np.linspace(0, n - 1, min_points)
        arr_interp = np.column_stack([
            np.interp(interp_idx, original_idx, col) for col in arr.T
        ])
        arr = arr_interp
        n = min_points
    else:
        interp_idx = original_idx  # no interpolation

    # Choose algorithm (Binseg is much faster than BottomUp for large n)
    algo = rpt.Binseg(model="l2").fit(arr)

    # Decide number of breakpoints
    if use_penalty:
        sigma = arr.std()
        penalty = np.log(n) * arr.shape[1] * sigma**2
        bkps_interp_idx = algo.predict(pen=penalty)
    else:
        bkps_interp_idx = algo.predict(n_bkps=int(np.ceil(n ** 0.5)))

    # Remove the last index (ruptures always adds len(arr))
    if bkps_interp_idx[-1] == n:
        bkps_interp_idx = bkps_interp_idx[:-1]

    # Map back to original indices
    bkps_original_idx = [int(round(interp_idx[i - 1])) for i in bkps_interp_idx]  # 1-based in ruptures

    # Remove duplicates while preserving order
    bkps_original_idx = np.unique(bkps_original_idx)

    return bkps_original_idx, bkps_interp_idx


def plot_analysis(arr_norm, bkps, save_path="video_analysis.png", max_points=1000):
    """Plot normalized metrics and detected change points (breakpoints) using index as x-axis."""
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(arr_norm.shape[0])
    colors = ["#2E86C1", "#E74C3C", "#28B463"]
    labels = ["GOP Bitrate", "I-frame Size", "GOP Variance"]

    # Downsample for faster plotting if too many points
    step = max(1, len(x) // max_points)

    for i in range(arr_norm.shape[1]):
        ax.plot(x[::step], arr_norm[::step, i], label=labels[i], color=colors[i], linewidth=2)

    # Draw change points as vertical lines
    for j, bkp in enumerate(bkps):
        ax.axvline(bkp, color='k', linestyle='--', alpha=0.7, label='Change Point' if j == 0 else None)

    ax.set_ylabel("Normalized Value")
    ax.set_xlabel("Frame Index")
    ax.set_title("Video Metrics & Change Points (Ruptures)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    print(f"Saved analysis: {save_path}")
    return fig


if __name__ == "__main__":
    try:
        # Analyze video and extract metrics
        results = analyze_video_metadata("./data/long.mp4")
        arr = np.column_stack((results["gop_bitrate"], results["i_frame_size"], results["gop_variances"]))

        # Normalize with optimized vectorized function
        arr_norm = normalize_array(arr)

        # Detect change points (optimized)
        bkps_original_idx, _ = detect_change_points(arr_norm, min_points=500, use_penalty=False)

        # Plot and save
        plot_analysis(arr_norm, bkps_original_idx)

    except FileNotFoundError:
        print("❌ Error: Video file not found")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
