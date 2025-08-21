import av
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

def analyze_video_metadata(video_path):
    """
    Fast video analysis using metadata only - adaptive to content patterns.
    """
    container = av.open(video_path)
    video_stream = None
    
    # Find video stream
    for stream in container.streams:
        if stream.type == "video":
            video_stream = stream
            break
    
    if not video_stream:
        raise ValueError("No video stream found")
    
    # Extract video properties
    fps = float(video_stream.average_rate) if video_stream.average_rate else 30.0
    width = video_stream.width or 1920
    height = video_stream.height or 1080
    
    print(f"Analyzing video: {width}x{height} @ {fps} fps")
    print("Processing metadata only...")
    
    # Fast data collection
    packet_data = []
    keyframe_data = []
    
    # GOP tracking
    gop_start_time = None
    gop_total_bits = 0
    gop_packet_count = 0
    gop_i_frame_size = 0
    
    # Process packets
    for packet in container.demux(video_stream):
        if packet.pts is None:
            continue
        
        timestamp = float(packet.pts * video_stream.time_base)
        size = packet.size
        
        packet_data.append((timestamp, size, packet.is_keyframe))
        
        if packet.is_keyframe:
            # Save previous GOP
            if gop_start_time is not None and gop_packet_count > 0:
                duration = timestamp - gop_start_time
                if duration > 0:
                    bitrate = gop_total_bits / duration
                    keyframe_data.append({
                        'time': gop_start_time,
                        'i_frame_size': gop_i_frame_size,
                        'gop_bitrate': bitrate,
                        'packet_count': gop_packet_count,
                        'duration': duration
                    })
            
            # Start new GOP
            gop_start_time = timestamp
            gop_total_bits = size * 8
            gop_packet_count = 1
            gop_i_frame_size = size
        else:
            gop_total_bits += size * 8
            gop_packet_count += 1
    
    # Handle last GOP
    if gop_start_time is not None and gop_packet_count > 0:
        if len(keyframe_data) > 0:
            avg_duration = np.mean([kf['duration'] for kf in keyframe_data])
        else:
            avg_duration = 2.0
        
        bitrate = gop_total_bits / avg_duration
        keyframe_data.append({
            'time': gop_start_time,
            'i_frame_size': gop_i_frame_size,
            'gop_bitrate': bitrate,
            'packet_count': gop_packet_count,
            'duration': avg_duration
        })
    
    container.close()
    
    if not packet_data or not keyframe_data:
        raise ValueError("No valid video data found")
    
    # Convert to arrays
    packet_times = np.array([p[0] for p in packet_data])
    packet_sizes = np.array([p[1] for p in packet_data])
    
    keyframe_times = np.array([kf['time'] for kf in keyframe_data])
    i_frame_sizes = np.array([kf['i_frame_size'] for kf in keyframe_data])
    gop_bitrates = np.array([kf['gop_bitrate'] for kf in keyframe_data])
    
    # Calculate variance with adaptive windowing (no scene change adaptation)
    variance_times, variance_values = calculate_variance_adaptive(
        packet_times, packet_sizes, [], keyframe_times[-1]
    )
    
    # Normalize metrics
    def normalize(arr):
        if len(arr) < 2 or np.ptp(arr) == 0:
            return np.zeros_like(arr)
        return (arr - np.min(arr)) / np.ptp(arr)
    
    # Core metrics
    heuristics = {
        "GOP Bitrate": normalize(gop_bitrates),
        "I-frame Size": normalize(i_frame_sizes),
        "Temporal Variance": normalize(variance_values)
    }
    
    times_dict = {
        "GOP Bitrate": keyframe_times,
        "I-frame Size": keyframe_times,
        "Temporal Variance": variance_times
    }
    
    # Calculate interest score with adaptive weighting (no scene change adaptation)
    interest_scores, interest_times = calculate_interest_adaptive(
        heuristics, times_dict, [], keyframe_times[-1]
    )
    
    # Statistics
    total_duration = packet_times[-1] if len(packet_times) > 0 else 0
    print(f"\nProcessing complete!")
    print(f"Duration: {int(total_duration//60):02d}:{int(total_duration%60):02d}")
    print(f"Total packets: {len(packet_data):,}")
    print(f"Total GOPs: {len(keyframe_data):,}")
    print(f"Avg GOP bitrate: {np.mean(gop_bitrates)/1000:.1f} kbps")
    
    return {
        'heuristics': heuristics,
        'times': times_dict,
        'interest_scores': interest_scores,
        'interest_times': interest_times,
        'stats': {
            'duration': total_duration,
            'packets': len(packet_data),
            'gops': len(keyframe_data),
            'fps': fps,
            'resolution': (width, height)
        }
    }

def detect_scene_changes_adaptive(times, i_frame_sizes, fps, resolution):
    """Adaptive scene change detection based on video characteristics."""
    if len(i_frame_sizes) < 10:
        return []
    
    # Initial conservative detection
    median_size = np.median(i_frame_sizes)
    mad = np.median(np.abs(i_frame_sizes - median_size)) * 1.4826  # Convert to std
    
    # Start conservative, then adapt
    initial_threshold = median_size + 4.0 * mad
    
    # Resolution factor
    pixels = resolution[0] * resolution[1]
    res_factor = (pixels / (1920 * 1080)) ** 0.3
    threshold = initial_threshold * res_factor
    
    # Find initial candidates
    candidates = []
    for i, (time, size) in enumerate(zip(times, i_frame_sizes)):
        if i > 2 and size > threshold:
            candidates.append(time)
    
    # Adaptive refinement based on initial detection
    if len(candidates) == 0:
        # Too conservative - relax threshold
        threshold = median_size + 2.5 * mad * res_factor
        for i, (time, size) in enumerate(zip(times, i_frame_sizes)):
            if i > 2 and size > threshold:
                candidates.append(time)
    
    # Calculate adaptive separation based on video duration and fps
    duration = times[-1] - times[0] if len(times) > 1 else 60
    base_separation = max(2.0, min(30.0, duration / 100))  # 2s to 30s based on length
    
    # If too many candidates, increase separation
    if len(candidates) > duration / 60 * 5:  # More than 5 per minute
        base_separation *= 2
    
    # Apply separation filter
    if len(candidates) <= 1:
        return candidates
    
    filtered = [candidates[0]]
    for time in candidates[1:]:
        if time - filtered[-1] >= base_separation:
            filtered.append(time)
    
    return filtered

def calculate_variance_adaptive(packet_times, packet_sizes, scene_changes, total_duration):
    """Calculate temporal variance with adaptive windowing."""
    if len(packet_times) < 10:
        return np.array([]), np.array([])
    
    # Adaptive window size based on scene change density
    scene_density = len(scene_changes) / max(total_duration / 60, 1)
    
    if scene_density < 0.5:  # Low activity content (like podcasts)
        n_windows = max(20, int(total_duration / 10))
    elif scene_density > 5:  # High activity content
        n_windows = max(50, int(total_duration / 2))
    else:  # Moderate activity
        n_windows = max(30, int(total_duration / 5))
    
    # Calculate variance in windows
    duration = packet_times[-1] - packet_times[0]
    window_size = duration / n_windows
    
    variance_times = []
    variance_values = []
    
    for i in range(n_windows):
        start_time = packet_times[0] + i * window_size
        end_time = start_time + window_size
        
        mask = (packet_times >= start_time) & (packet_times < end_time)
        window_sizes = packet_sizes[mask]
        
        if len(window_sizes) > 5:
            variance_times.append(start_time + window_size/2)
            variance_values.append(np.var(window_sizes))
    
    return np.array(variance_times), np.array(variance_values)

def calculate_interest_adaptive(heuristics, times_dict, scene_changes, total_duration):
    """Calculate interest score with adaptive weighting based on content patterns."""
    
    # Determine content characteristics from scene changes
    scene_density = len(scene_changes) / max(total_duration / 60, 1)
    
    # Adaptive weights based on observed patterns
    if scene_density < 0.5:  # Static content (podcasts, lectures)
        weights = {
            "GOP Bitrate": 0.5,
            "Temporal Variance": 0.3,
            "I-frame Size": 0.2
        }
    elif scene_density > 3:  # Dynamic content (movies, sports)
        weights = {
            "I-frame Size": 0.5,
            "GOP Bitrate": 0.3,
            "Temporal Variance": 0.2
        }
    else:  # Moderate content
        weights = {
            "GOP Bitrate": 0.4,
            "I-frame Size": 0.35,
            "Temporal Variance": 0.25
        }
    
    # Use most common timeline as reference
    ref_key = max(times_dict.keys(), key=lambda k: len(times_dict[k]))
    ref_times = times_dict[ref_key]
    
    if len(ref_times) == 0:
        return np.array([]), np.array([])
    
    scores = np.zeros(len(ref_times))
    
    for metric, weight in weights.items():
        if metric in heuristics and len(heuristics[metric]) > 0:
            metric_times = times_dict[metric]
            metric_values = heuristics[metric]
            
            if len(metric_values) == len(ref_times):
                scores += weight * metric_values
            elif len(metric_times) > 1:
                interpolated = np.interp(ref_times, metric_times, metric_values)
                scores += weight * interpolated
    
    return scores, ref_times

def plot_analysis(results, save_path="video_analysis.png"):
    """Clean visualization focused on interest score."""
    heuristics = results['heuristics']
    times_dict = results['times']
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    # Plot 1: Individual metrics
    ax1 = axes[0]
    colors = ['#2E86C1', '#E74C3C', '#28B463', '#F39C12']
    for i, (name, values) in enumerate(heuristics.items()):
        if len(values) > 0:
            times = times_dict[name]
            color = colors[i % len(colors)]
            ax1.plot(times, values, label=name, alpha=0.8, color=color, linewidth=1.5)
    ax1.set_ylabel("Normalized Values")
    ax1.set_title("Analysis Metrics")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    # Plot 2: Interest score - main focus
    ax2 = axes[1]
    # Smooth interest score using moving average (window=5)
    def smooth(arr, window=5):
        if len(arr) < window:
            return arr
        return np.convolve(arr, np.ones(window)/window, mode='same')
    smoothed_interest_scores = smooth(results['interest_scores'], window=5)
    interest_times = results['interest_times']
    if len(smoothed_interest_scores) > 0:
        ax2.plot(interest_times, smoothed_interest_scores, color='#8E44AD', linewidth=2.5, label='Interest Score (Smoothed)')
        ax2.fill_between(interest_times, smoothed_interest_scores, alpha=0.4, color='#8E44AD')
        max_time = interest_times[-1] if len(interest_times) > 0 else 100
        ax2.set_xlim(0, max_time)
        if max_time > 3600:
            major_ticks = np.arange(0, max_time, 900)
            ax2.set_xticks(major_ticks)
            ax2.set_xticklabels([f"{int(t//3600)}:{int((t%3600)//60):02d}:{int(t%60):02d}" for t in major_ticks])
        else:
            major_ticks = np.arange(0, max_time, 60)
            ax2.set_xticks(major_ticks)
            ax2.set_xticklabels([f"{int(t//60):02d}:{int(t%60):02d}" for t in major_ticks])
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Interest Score")
    ax2.set_title("Interest Score - Main Reference", fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    print(f"Saved analysis: {save_path}")
    return fig

if __name__ == "__main__":
    import time
    
    try:
        print("🚀 Fast Video Analysis - Adaptive & Clean")
        start_time = time.time()
        
        # Analyze video
        results = analyze_video_metadata("./data/test2.mp4")
        
        processing_time = time.time() - start_time
        print(f"⚡ Processing completed in {processing_time:.2f} seconds")
        
        # Create visualization
        plot_analysis(results)
        
        
    except FileNotFoundError:
        print("❌ Error: Video file not found")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()