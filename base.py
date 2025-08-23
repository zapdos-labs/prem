from utils import timeit_log
import av
import numpy as np
import numba as nb
import multiprocessing as mp
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@timeit_log
def analyze_video_metadata(video_path):
    """
    Fast video analysis using metadata only, fully optimized with NumPy and Numba.
    
    Returns a dict of:
      - gop_bitrate: NumPy array of GOP bitrates
      - i_frame_size: NumPy array of I-frame sizes
      - gop_variances: NumPy array of GOP size variances
      - i_frame_times: NumPy array of I-frame timestamps
    """
    logger.info(f"Starting video analysis for: {video_path}")
    
    # Container setup
    thread_count = min(4, mp.cpu_count())
    container = av.open(video_path, options={
        'threads': str(thread_count),
        'thread_type': 'frame',
        'buffer_size': '4096000',  # 4MB
        'analyzeduration': '1000000',
        'probesize': '1000000'
    })
    video_stream = next((s for s in container.streams if s.type == "video"), None)
    if not video_stream:
        container.close()
        raise ValueError("No video stream found")
    
    fps = float(video_stream.average_rate) if video_stream.average_rate else 30.0
    time_base = float(video_stream.time_base)

    # --- Step 1: Read packets ---
    estimated_duration = video_stream.duration * video_stream.time_base if video_stream.duration else 1200
    estimated_packets = int(estimated_duration * fps * 1.2)
    chunk_size = 25000 if estimated_packets < 100000 else 50000

    all_timestamps, all_sizes, all_keyframes = [], [], []
    timestamps_chunk = np.empty(chunk_size, dtype=np.float64)
    sizes_chunk = np.empty(chunk_size, dtype=np.int32)
    keyframes_chunk = np.empty(chunk_size, dtype=bool)
    
    chunk_count = 0
    count = 0
    
    try:
        valid_packets = (
            (packet.pts * time_base, packet.size, packet.is_keyframe)
            for packet in container.demux(video_stream) if packet.pts is not None
        )
        for pts_time, size, is_keyframe in valid_packets:
            timestamps_chunk[chunk_count] = pts_time
            sizes_chunk[chunk_count] = size
            keyframes_chunk[chunk_count] = is_keyframe
            chunk_count += 1
            
            if chunk_count == chunk_size:
                all_timestamps.append(timestamps_chunk.copy())
                all_sizes.append(sizes_chunk.copy())
                all_keyframes.append(keyframes_chunk.copy())
                count += chunk_size
                chunk_count = 0
        
        if chunk_count > 0:
            all_timestamps.append(timestamps_chunk[:chunk_count].copy())
            all_sizes.append(sizes_chunk[:chunk_count].copy())
            all_keyframes.append(keyframes_chunk[:chunk_count].copy())
            count += chunk_count
    finally:
        container.close()
    
    if count == 0:
        logger.warning("No valid packets found")
        return {"gop_bitrate": np.array([]), "i_frame_size": np.array([]), "gop_variances": np.array([]), "i_frame_times": np.array([])}
    
    timestamps = np.concatenate(all_timestamps)
    sizes = np.concatenate(all_sizes)
    keyframes = np.concatenate(all_keyframes)
    
    # --- Step 2: Keyframe indices ---
    keyframe_indices = np.flatnonzero(keyframes)
    if len(keyframe_indices) == 0:
        logger.warning("No keyframes found")
        return {"gop_bitrate": np.array([]), "i_frame_size": np.array([]), "gop_variances": np.array([]), "i_frame_times": np.array([])}
    
    i_frame_times = timestamps[keyframe_indices]
    
    # --- Step 3: Compute GOP metrics using Numba (parallel) ---
    gop_bitrates, i_frame_sizes, gop_variances = _compute_gop_metrics_numba(sizes, timestamps, keyframe_indices, fps)
    
    return {
        "gop_bitrate": gop_bitrates,
        "i_frame_size": i_frame_sizes,
        "gop_variances": gop_variances,
        "i_frame_times": i_frame_times
    }

@nb.njit(parallel=True, fastmath=True, cache=True)
def _compute_gop_metrics_numba(sizes, timestamps, keyframe_indices, fps):
    n_gops = len(keyframe_indices)
    gop_bitrates = np.empty(n_gops, dtype=np.float64)
    i_frame_sizes = np.empty(n_gops, dtype=np.int32)
    gop_variances = np.empty(n_gops, dtype=np.float64)
    
    for i in nb.prange(n_gops):
        start_idx = keyframe_indices[i]
        end_idx = keyframe_indices[i+1] if i+1 < n_gops else len(sizes)
        gop_length = end_idx - start_idx
        
        i_frame_sizes[i] = sizes[start_idx]
        gop_size_sum = 0
        for j in range(start_idx, end_idx):
            gop_size_sum += sizes[j]
        
        gop_duration = timestamps[end_idx-1] - timestamps[start_idx] if gop_length > 0 else 1.0/fps
        if gop_duration <= 0:
            gop_duration = 1.0 / fps
        
        gop_bitrates[i] = (gop_size_sum * 8.0) / gop_duration
        
        if gop_length > 1:
            mean = gop_size_sum / gop_length
            variance_sum = 0.0
            for j in range(start_idx, end_idx):
                diff = sizes[j] - mean
                variance_sum += diff * diff
            gop_variances[i] = variance_sum / (gop_length - 1)
        else:
            gop_variances[i] = 0.0
    
    return gop_bitrates, i_frame_sizes, gop_variances
