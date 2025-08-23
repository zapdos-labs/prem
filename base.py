from utils import timeit_log
import av
import numpy as np
import numba as nb
from concurrent.futures import ThreadPoolExecutor
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
    start_time = time.perf_counter()
    
    # Container setup with optimized threading and buffering
    thread_count = min(4, mp.cpu_count())
    logger.info(f"Opening container with {thread_count} threads")
    
    container_start = time.perf_counter()
    container = av.open(video_path, options={
        'threads': str(thread_count),
        'thread_type': 'frame',
        'buffer_size': '4096000',  # 4MB buffer for faster I/O
        'analyzeduration': '1000000',  # Reduce analysis time
        'probesize': '1000000'  # Reduce probe size for faster opening
    })
    video_stream = next((s for s in container.streams if s.type == "video"), None)
    if not video_stream:
        container.close()
        raise ValueError("No video stream found")
    
    container_time = time.perf_counter() - container_start
    logger.info(f"Container opened in {container_time:.3f}s")
    
    fps = float(video_stream.average_rate) if video_stream.average_rate else 30.0
    width = video_stream.width or 1920
    height = video_stream.height or 1080
    logger.info(f"Video specs: {width}x{height} @ {fps:.2f} fps")
    logger.info(f"Video duration: {video_stream.duration * video_stream.time_base:.2f}s")

    # --- Step 1: Optimized packet reading with adaptive chunking ---
    # Estimate total packets for better memory allocation
    estimated_duration = video_stream.duration * video_stream.time_base if video_stream.duration else 1200
    estimated_packets = int(estimated_duration * fps * 1.2)  # 20% buffer
    
    # Adaptive chunk size based on estimated packets
    if estimated_packets < 10000:
        chunk_size = estimated_packets
    elif estimated_packets < 100000:
        chunk_size = 25000
    else:
        chunk_size = 50000
    
    logger.info(f"Estimated {estimated_packets:,} packets, using chunk size: {chunk_size:,}")
    
    all_timestamps = []
    all_sizes = []
    all_keyframes = []
    
    # Pre-allocate chunk buffers
    timestamps_chunk = np.empty(chunk_size, dtype=np.float64)
    sizes_chunk = np.empty(chunk_size, dtype=np.int32)
    keyframes_chunk = np.empty(chunk_size, dtype=bool)
    
    count = 0
    chunk_count = 0
    chunks_processed = 0
    time_base = float(video_stream.time_base)  # Cache time_base conversion
    logger.info(f"Cached time_base: {time_base}")
    
    # Batch packet processing with minimal Python overhead
    demux_start = time.perf_counter()
    last_log_time = demux_start
    
    try:
        # Optimized single-pass demuxing with pre-computed batch handling
        logger.info("Using optimized container demux with minimal overhead")
        
        # Use generator expression for minimal memory allocation
        valid_packets = (
            (packet.pts * time_base, packet.size, packet.is_keyframe)
            for packet in container.demux(video_stream) 
            if packet.pts is not None
        )
        
        # Process packets with minimal Python call overhead
        for pts_time, size, is_keyframe in valid_packets:
            timestamps_chunk[chunk_count] = pts_time
            sizes_chunk[chunk_count] = size
            keyframes_chunk[chunk_count] = is_keyframe
            chunk_count += 1
            
            # Chunk boundary check - optimized for hot path
            if chunk_count == chunk_size:
                all_timestamps.append(timestamps_chunk.copy())
                all_sizes.append(sizes_chunk.copy())  
                all_keyframes.append(keyframes_chunk.copy())
                count += chunk_size
                chunk_count = 0
                chunks_processed += 1
                
                # Progress logging every 2 seconds
                current_time = time.perf_counter()
                if current_time - last_log_time > 2.0:
                    rate = count / (current_time - demux_start)
                    logger.info(f"Processed {count:,} packets - {rate:,.0f} packets/sec")
                    last_log_time = current_time
        
        # Handle remaining packets in final partial chunk
        if chunk_count > 0:
            all_timestamps.append(timestamps_chunk[:chunk_count].copy())
            all_sizes.append(sizes_chunk[:chunk_count].copy())
            all_keyframes.append(keyframes_chunk[:chunk_count].copy())
            count += chunk_count
            chunks_processed += 1
            
    finally:
        container.close()
    
    demux_time = time.perf_counter() - demux_start
    logger.info(f"Demuxing completed: {count:,} packets in {chunks_processed} chunks ({demux_time:.3f}s)")
    logger.info(f"Demux performance: {count/demux_time:,.0f} packets/sec")

    if count == 0:
        logger.warning("No valid packets found in video stream")
        return {
            "gop_bitrate": np.array([]),
            "i_frame_size": np.array([]),
            "gop_variances": np.array([]),
            "i_frame_times": np.array([])
        }

    # Concatenate all chunks efficiently
    concat_start = time.perf_counter()
    timestamps = np.concatenate(all_timestamps) if all_timestamps else np.array([])
    sizes = np.concatenate(all_sizes) if all_sizes else np.array([])
    keyframes = np.concatenate(all_keyframes) if all_keyframes else np.array([])
    concat_time = time.perf_counter() - concat_start
    logger.info(f"Array concatenation completed in {concat_time:.3f}s")

    # --- Step 2: Find keyframe indices more efficiently ---
    keyframe_start = time.perf_counter()
    keyframe_indices = np.flatnonzero(keyframes)  # More efficient than np.where()[0]
    keyframe_time = time.perf_counter() - keyframe_start
    
    logger.info(f"Found {len(keyframe_indices):,} keyframes out of {count:,} packets ({len(keyframe_indices)/count*100:.1f}%)")
    logger.info(f"Keyframe detection completed in {keyframe_time:.3f}s")
    
    if len(keyframe_indices) == 0:
        logger.warning("No keyframes found in video stream")
        return {
            "gop_bitrate": np.array([]),
            "i_frame_size": np.array([]),
            "gop_variances": np.array([]),
            "i_frame_times": np.array([])
        }
    
    i_frame_times = timestamps[keyframe_indices]
    n_gops = len(keyframe_indices)
    avg_gop_size = count // n_gops if n_gops > 0 else 0
    logger.info(f"Analysis will process {n_gops:,} GOPs (avg {avg_gop_size:.1f} frames per GOP)")

    # --- Step 3: Optimized GOP metrics computation ---
    metrics_start = time.perf_counter()
    result = _compute_gop_metrics_parallel(sizes, timestamps, keyframe_indices, fps, i_frame_times)
    metrics_time = time.perf_counter() - metrics_start
    logger.info(f"GOP metrics computation completed in {metrics_time:.3f}s")
    
    total_time = time.perf_counter() - start_time
    logger.info(f"Total analysis completed in {total_time:.3f}s")
    logger.info(f"Performance: {count/total_time:,.0f} packets/sec, {n_gops/total_time:,.0f} GOPs/sec")
    
    return result

@nb.njit(parallel=True, fastmath=True, cache=True)
def _compute_gop_metrics_numba(sizes, timestamps, keyframe_indices, fps):
    """Numba-optimized GOP metrics computation with manual vectorization."""
    n_gops = len(keyframe_indices)
    gop_bitrates = np.empty(n_gops, dtype=np.float64)
    i_frame_sizes = np.empty(n_gops, dtype=np.int32)
    gop_variances = np.empty(n_gops, dtype=np.float64)
    
    # Parallel processing of GOPs
    for i in nb.prange(n_gops):
        start_idx = keyframe_indices[i]
        end_idx = keyframe_indices[i+1] if i+1 < n_gops else len(sizes)
        
        # Extract GOP data
        gop_size_sum = 0
        gop_length = end_idx - start_idx
        i_frame_sizes[i] = sizes[start_idx]
        
        # Calculate sum in one pass
        for j in range(start_idx, end_idx):
            gop_size_sum += sizes[j]
        
        # Calculate duration
        if end_idx > start_idx:
            gop_duration = timestamps[end_idx-1] - timestamps[start_idx]
            if gop_duration <= 0:
                gop_duration = 1.0 / fps
        else:
            gop_duration = 1.0 / fps
        
        # Calculate bitrate
        gop_bitrates[i] = (gop_size_sum * 8.0) / gop_duration
        
        # Calculate variance efficiently
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

def _compute_gop_metrics_parallel(sizes, timestamps, keyframe_indices, fps, i_frame_times):
    """Wrapper function that decides between parallel and sequential processing."""
    n_gops = len(keyframe_indices)
    n_cores = min(4, mp.cpu_count())
    
    logger.info(f"Computing metrics for {n_gops:,} GOPs using {n_cores} cores")
    
    # Use parallel processing for larger datasets
    if n_gops > 1000:
        chunk_size = max(250, n_gops // n_cores)
        logger.info(f"Large dataset detected. Chunk size: {chunk_size}")
        
        if n_gops > chunk_size * 2:  # Only parallelize if worthwhile
            logger.info("Using ThreadPoolExecutor for parallel GOP processing")
            return _parallel_gop_processing(sizes, timestamps, keyframe_indices, fps, i_frame_times, n_cores)
        else:
            logger.info("Dataset not large enough for effective parallelization, using Numba")
    else:
        logger.info("Small dataset, using optimized Numba processing")
    
    # Use optimized numba for smaller datasets or when parallelization isn't beneficial
    numba_start = time.perf_counter()
    gop_bitrates, i_frame_sizes, gop_variances = _compute_gop_metrics_numba(
        sizes, timestamps, keyframe_indices, fps
    )
    numba_time = time.perf_counter() - numba_start
    logger.info(f"Numba processing completed in {numba_time:.3f}s")
    
    return {
        "gop_bitrate": gop_bitrates,
        "i_frame_size": i_frame_sizes,
        "gop_variances": gop_variances,
        "i_frame_times": i_frame_times
    }

def _parallel_gop_processing(sizes, timestamps, keyframe_indices, fps, i_frame_times, n_cores):
    """Parallel processing of GOP metrics using ThreadPoolExecutor."""
    n_gops = len(keyframe_indices)
    chunk_size = (n_gops + n_cores - 1) // n_cores  # Ceiling division
    
    logger.info(f"Parallel processing: {n_gops} GOPs across {n_cores} workers, {chunk_size} GOPs per chunk")
    
    def process_chunk(start_gop, end_gop):
        chunk_start = time.perf_counter()
        chunk_indices = keyframe_indices[start_gop:end_gop]
        result = _compute_gop_metrics_numba(sizes, timestamps, chunk_indices, fps)
        chunk_time = time.perf_counter() - chunk_start
        logger.debug(f"Processed chunk [{start_gop}:{end_gop}] ({end_gop-start_gop} GOPs) in {chunk_time:.3f}s")
        return result
    
    # Create chunks
    chunks = [(i, min(i + chunk_size, n_gops)) for i in range(0, n_gops, chunk_size)]
    logger.info(f"Created {len(chunks)} chunks for parallel processing")
    
    # Process chunks in parallel
    parallel_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_cores) as executor:
        results = list(executor.map(lambda chunk: process_chunk(*chunk), chunks))
    parallel_time = time.perf_counter() - parallel_start
    
    logger.info(f"Parallel processing completed in {parallel_time:.3f}s")
    
    # Combine results
    combine_start = time.perf_counter()
    if results:
        gop_bitrates = np.concatenate([r[0] for r in results])
        i_frame_sizes = np.concatenate([r[1] for r in results])
        gop_variances = np.concatenate([r[2] for r in results])
        logger.info(f"Combined {len(results)} result chunks")
    else:
        gop_bitrates = np.array([])
        i_frame_sizes = np.array([])
        gop_variances = np.array([])
        logger.warning("No results from parallel processing")
    
    combine_time = time.perf_counter() - combine_start
    logger.info(f"Result combination completed in {combine_time:.3f}s")
    
    return {
        "gop_bitrate": gop_bitrates,
        "i_frame_size": i_frame_sizes,
        "gop_variances": gop_variances,
        "i_frame_times": i_frame_times
    }