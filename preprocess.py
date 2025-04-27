import os
import cv2
from moviepy.editor import VideoFileClip
import torch
import whisper

# Whisper ASR模型只加载一次，加速！
whisper_model = whisper.load_model("large")

def extract_middle_frame(video_path: str, output_image_path: str) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频文件：{video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise RuntimeError(f"无法获取视频总帧数：{video_path}")

    mid_idx = total_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"读取第 {mid_idx} 帧失败：{video_path}")

    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    cv2.imwrite(output_image_path, frame)


def extract_audio(video_path: str, output_audio_path: str) -> None:
    clip = VideoFileClip(video_path)
    if clip.audio is None:
        raise RuntimeError(f"视频中没有音频轨道：{video_path}")
    os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)
    clip.audio.write_audiofile(output_audio_path, codec='pcm_s16le', verbose=False, logger=None)


def extract_text_from_audio(audio_path: str, output_text_path: str) -> None:
    result = whisper_model.transcribe(audio_path)
    os.makedirs(os.path.dirname(output_text_path), exist_ok=True)
    with open(output_text_path, "w", encoding="utf-8") as f:
        f.write(result['text'])


if __name__ == "__main__":
    # 输入数据集根目录
    input_root = "E:/New_project/MiniGPT-4/CAER_validation"
    output_image_root = "E:/New_project/MiniGPT-4/extract_data/CAER_validation_image"
    output_audio_root = "E:/New_project/MiniGPT-4/extract_data/CAER_validation_audio"
    output_asr_root   = "E:/New_project/MiniGPT-4/extract_data/CAER_validation_asr"

    for emotion_folder in os.listdir(input_root):
        input_emotion_path = os.path.join(input_root, emotion_folder)

        if not os.path.isdir(input_emotion_path):
            continue  # 跳过不是文件夹的内容

        for filename in os.listdir(input_emotion_path):
            if not filename.endswith(".avi"):
                continue

            video_path = os.path.join(input_emotion_path, filename)

            # 输出路径
            frame_path = os.path.join(output_image_root, emotion_folder, filename.replace(".avi", ".jpg"))
            audio_path = os.path.join(output_audio_root, emotion_folder, filename.replace(".avi", ".wav"))
            text_path  = os.path.join(output_asr_root, emotion_folder, filename.replace(".avi", ".txt"))

            print(f"正在处理 {video_path} ...")

            try:
                extract_middle_frame(video_path, frame_path)
                extract_audio(video_path, audio_path)
                extract_text_from_audio(audio_path, text_path)
                torch.cuda.empty_cache()  # 清理CUDA缓存，避免OOM
            except Exception as e:
                print(f"处理出错 {video_path} ：{e}")

    print("全部预处理完成 ✅")
