import av
import cv2
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count

def save_frame(frame, ts_ms, output_dir):
    """Convert frame to BGR and save as JPEG."""
    img_bgr = cv2.cvtColor(frame.to_ndarray(format='rgb24'), cv2.COLOR_RGB2BGR)
    m, s, ms_part = ts_ms // 60000, (ts_ms % 60000) // 1000, ts_ms % 1000
    filename = f"frame_{m:03d}m_{s:02d}s_{ms_part:03d}ms.jpg"
    cv2.imwrite(os.path.join(output_dir, filename), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

def process_chunk(args):
    video_path, chunk_timestamps, output_dir, device = args
    # Open container
    container = av.open(video_path)
    
    if device == "gpu":
        try:
            container.close()
            container = av.open(video_path, options={'hwaccel': 'auto'})
            print("Worker using GPU acceleration")
        except Exception:
            container = av.open(video_path)
            print("Worker GPU failed, using CPU")

    stream = container.streams.video[0]

    for ts_ms in chunk_timestamps:
        try:
            # Seek to nearest keyframe
            seek_target = int(ts_ms * av.time_base / 1000)
            container.seek(seek_target, any_frame=False, stream=stream)

            # Decode first frame (keyframe)
            for frame in container.decode(stream):
                if frame.pts is not None:
                    save_frame(frame, ts_ms, output_dir)
                    break
        except Exception as e:
            print(f"Error at {ts_ms}ms: {e}")

    container.close()
    return len(chunk_timestamps)

def open_video(video_path, device="cpu"):
    container = av.open(video_path)
    if device == "gpu":
        try:
            container.close()
            container = av.open(video_path, options={'hwaccel': 'auto'})
            print("Using GPU acceleration")
        except Exception:
            container = av.open(video_path)
            print("GPU acceleration failed, using CPU")

    stream = container.streams.video[0]
    fps = float(stream.average_rate)
    duration_ms = float(stream.duration * stream.time_base * 1000)
    container.close()
    return fps, duration_ms

def generate_timestamps(duration_ms, interval_ms):
    return list(range(0, int(duration_ms), int(interval_ms)))

def parallel_extract(video_path, timestamps_ms, output_dir, device="gpu", workers=None):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    if workers is None:
        workers = min(max(1, cpu_count() - 1), 4)

    chunk_size = (len(timestamps_ms) + workers - 1) // workers
    chunks = [timestamps_ms[i:i + chunk_size] for i in range(0, len(timestamps_ms), chunk_size)]
    args = [(video_path, chunk, output_dir, device) for chunk in chunks]

    with Pool(workers) as pool:
        counts = pool.map(process_chunk, args)
    print(f"Done! Extracted {sum(counts)} frames to '{output_dir}'")

# -----------------------------
if __name__ == "__main__":
    video_path = "./data/long.mp4"
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    _, duration_ms = open_video(video_path, device="gpu")
    timestamps = generate_timestamps(duration_ms, interval_ms=30000)  # every 30s
    parallel_extract(video_path, timestamps, "./data/frames_gpu_30s", device="gpu")
