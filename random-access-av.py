import av
import cv2
import os
from pathlib import Path

def open_video(video_path, device="cpu"):
    """Open video container with optional GPU acceleration and return container + stream info."""
    container = av.open(video_path)
    stream = container.streams.video[0]

    if device == "gpu":
        try:
            container.close()
            container = av.open(video_path, options={'hwaccel': 'auto'})
            stream = container.streams.video[0]
            print("Using GPU acceleration")
        except Exception:
            container = av.open(video_path)
            stream = container.streams.video[0]
            print("GPU acceleration failed, using CPU")

    fps = float(stream.average_rate)
    duration_ms = float(stream.duration * stream.time_base * 1000)
    print(f"Video: {duration_ms:.0f}ms, FPS: {fps:.2f}, Resolution: {stream.width}x{stream.height}, Device: {device}")
    return container, stream, fps, duration_ms

def generate_timestamps(duration_ms, interval_ms=None, timestamps_ms=None):
    """Generate extraction timestamps in milliseconds."""
    if timestamps_ms is not None:
        return [ts for ts in timestamps_ms if ts <= duration_ms]
    if interval_ms is None:
        raise ValueError("Either interval_ms or timestamps_ms must be provided")
    return list(range(0, int(duration_ms), int(interval_ms)))

def save_frame(frame, timestamp_ms, output_dir):
    """Save a single frame as JPEG. Timestamp only used for filename."""
    img_bgr = cv2.cvtColor(frame.to_ndarray(format='rgb24'), cv2.COLOR_RGB2BGR)
    m, s, ms_part = timestamp_ms // 60000, (timestamp_ms % 60000) // 1000, timestamp_ms % 1000
    filename = f"frame_{m:03d}m_{s:02d}s_{ms_part:03d}ms.jpg"
    cv2.imwrite(os.path.join(output_dir, filename), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

def extract_frames(video_path, output_dir, timestamps_ms, container, stream, fps):
    """Extract frames at given timestamps efficiently using PyAV."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    total = len(timestamps_ms)

    for i, ts_ms in enumerate(timestamps_ms, 1):
        try:
            # Compute PTS once
            target_pts = int(ts_ms / (stream.time_base * 1000))
            seek_target = int(ts_ms * av.time_base / 1000)
            container.seek(seek_target, backward=True, stream=stream)

            best_frame, min_diff, frames_checked = None, float('inf'), 0
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue

                diff = abs(frame.pts - target_pts)
                if diff < min_diff:
                    best_frame, min_diff = frame, diff

                frames_checked += 1
                # Early exit: close enough or enough frames checked
                if diff <= fps / 2 or frames_checked >= fps * 2:
                    break
                if frame.pts > target_pts + fps * 2:
                    break

            if best_frame:
                save_frame(best_frame, ts_ms, output_dir)
                if i % 10 == 0 or i == total:
                    print(f"Extracted {i}/{total} frames")
            else:
                print(f"Warning: Could not extract frame at {ts_ms}ms")

        except Exception as e:
            print(f"Error extracting frame at {ts_ms}ms: {e}")

    container.close()
    print(f"Done! Extracted {total} frames to '{output_dir}'")

if __name__ == "__main__":
    video_path = "./data/long.mp4"
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    # --- Open video once ---
    container, stream, fps, duration_ms = open_video(video_path, device="gpu")

    # --- Generate timestamps in milliseconds (e.g., every 30s = 30000ms) ---
    timestamps = generate_timestamps(duration_ms, interval_ms=30000)

    # --- Extract frames efficiently ---
    extract_frames(video_path, "./data/frames_gpu_30s", timestamps, container, stream, fps)

    # --- Example for specific timestamps ---
    # specific_times = [60000, 180000, 300000, 480000, 600000]
    # container, stream, fps, duration_ms = open_video(video_path, device="cpu")
    # timestamps = generate_timestamps(duration_ms, timestamps_ms=specific_times)
    # extract_frames(video_path, "frames_specific", timestamps, container, stream, fps)
