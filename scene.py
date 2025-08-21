from scenedetect import VideoManager, SceneManager, open_video
from scenedetect.detectors import AdaptiveDetector
from scenedetect.video_splitter import split_video_ffmpeg

def detect_podcast_scenes(video_path, frame_skip=60 * 30):
    """
    Fast scene detection for podcast-style videos.
    """
    scene_manager = SceneManager()
    scene_manager.add_detector(AdaptiveDetector())

    # Pass frame_skip here (new API)
    print('Open video')
    video = open_video(video_path)
    print('Start')
    scene_manager.detect_scenes(video=video, frame_skip=frame_skip)

    # Retrieve scene list
    scene_list = scene_manager.get_scene_list()
    print(f"✅ Detected {len(scene_list)} scenes in {video_path}")

    # if scene_list:
    #     split_video_ffmpeg(video_path, scene_list)

    return scene_list


if __name__ == "__main__":
    scenes = detect_podcast_scenes("./data/long.mp4")
    print(scenes)
