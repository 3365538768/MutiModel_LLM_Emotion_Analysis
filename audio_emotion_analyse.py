from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
import subprocess
torch.manual_seed(1234)
import os

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-Audio-Chat",
                                          trust_remote_code=True,
                                          cache_dir="E:/New_project/MiniGPT-4/checkpoints/Qwen",
                                          )

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-Audio-Chat",
                                             device_map="cuda",
                                             trust_remote_code=True,
                                             cache_dir="E:/New_project/MiniGPT-4/checkpoints/Qwen",).eval()
def get_audio_response(audio_path,audio_question):
    # Note: The default behavior now has injection attack prevention off.
    query = tokenizer.from_list_format([
        {'audio': audio_path}, # Either a local path or an url
        {'text': audio_question},
    ])
    response, history = model.chat(tokenizer, query=query, history=None)
    return response

def extract_audio(video_path, output_audio_path):
    """
    用ffmpeg从视频中提取音频
    """
    command = [
        "ffmpeg",
        "-y",  # 覆盖已有文件
        "-i", video_path,
        "-ar", "16000",  # 采样率16k
        "-ac", "1",      # 单声道
        "-vn",           # 去掉视频流
        output_audio_path
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

if __name__ == "__main__":
    # 你的数据集路径
    input_root = "E:/New_project/MiniGPT-4/CAER_validation"
    output_root = "E:/New_project/MiniGPT-4/extract_data/CAER_validation_audio_text"
    os.makedirs(output_root, exist_ok=True)

    # 要问的问题
    audio_question = "what is the tone and emotion of the person? Give your reasons and describe it as detailed as possible."

    # 遍历每个情感类别子文件夹
    for emotion_folder in os.listdir(input_root):
        input_emotion_path = os.path.join(input_root, emotion_folder)
        output_emotion_path = os.path.join(output_root, emotion_folder)
        os.makedirs(output_emotion_path, exist_ok=True)

        # 遍历每个视频
        for filename in os.listdir(input_emotion_path):
            if filename.endswith(".avi"):
                video_path = os.path.join(input_emotion_path, filename)
                audio_path = os.path.join(output_emotion_path, filename.replace(".avi", ".wav"))
                txt_path = os.path.join(output_emotion_path, filename.replace(".avi", ".txt"))

                print(f"Processing {video_path}...")

                # 1. 提取音频
                extract_audio(video_path, audio_path)

                # 2. 获取音频情感分析
                try:
                    response = get_audio_response(audio_path, audio_question)
                except Exception as e:
                    print(f"Error processing {video_path}: {e}")
                    continue

                # 3. 保存输出文本
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(response)

                # 4. 删除临时音频文件（可选，看你是否要保留）
                os.remove(audio_path)

    print("全部完成！")