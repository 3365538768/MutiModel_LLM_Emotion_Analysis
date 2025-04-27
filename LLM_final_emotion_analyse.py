from transformers import LlamaForCausalLM, LlamaTokenizer
import torch
import os
# 指向你本地的模型路径
model_path = "E:/New_project/MiniGPT-4/checkpoints/Llama-2-7b-chat-hf"

# 加载分词器
tokenizer = LlamaTokenizer.from_pretrained(model_path)

# 加载模型（如果有 GPU，最好加上 device_map）
model = LlamaForCausalLM.from_pretrained(
    model_path,
    load_in_4bit=True,    # 如果GPU显存足够，可以用float16更快
    device_map="auto",             # 自动分配到GPU
)

# 模型设为eval模式
model.eval()

def ask(question):
    inputs = tokenizer(question, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=256)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def read_txt(path):
    with open (path, 'r') as f:
        information=f.read()
    return information

if __name__ == "__main__":
    # 输入路径（分别是前面处理得到的3个目录）
    asr_root = "E:/New_project/MiniGPT-4/extract_data/CAER-validation-asr"
    img_msg_root = "E:/New_project/MiniGPT-4/extract_data/CAER-validation-image-text"
    voice_msg_root = "E:/New_project/MiniGPT-4/extract_data/CAER-validation-audio-text"  # 注意这个路径是音频情感分析得到的文本
    # （如果之前命名不同，请改成你的实际audio-text保存路径）

    output_root = "E:/New_project/MiniGPT-4/extract_data/CAER-validation-LLM-text"
    os.makedirs(output_root, exist_ok=True)

    for emotion_folder in os.listdir(asr_root):
        input_asr_path = os.path.join(asr_root, emotion_folder)
        input_img_path = os.path.join(img_msg_root, emotion_folder)
        input_voice_path = os.path.join(voice_msg_root, emotion_folder)
        output_emotion_path = os.path.join(output_root, emotion_folder)
        os.makedirs(output_emotion_path, exist_ok=True)

        if not os.path.isdir(input_asr_path):
            continue  # 不是文件夹的跳过

        for filename in os.listdir(input_asr_path):
            if not filename.endswith(".txt"):
                continue

            # 三个文件路径
            asr_path = os.path.join(input_asr_path, filename)
            img_msg_path = os.path.join(input_img_path, filename)
            voice_msg_path = os.path.join(input_voice_path, filename)

            output_txt_path = os.path.join(output_emotion_path, filename)

            print(f"Processing {filename}...")

            try:
                # 读取三种信息
                asr_text = read_txt(asr_path)
                image_message = read_txt(img_msg_path)
                voice_message = read_txt(voice_msg_path)

                # 拼问题
                full_question = (
                    f"What is the person's emotion based on the following information? "
                    f"Information1: {asr_text}; "
                    f"Information2: {image_message}; "
                    f"Information3: {voice_message}"
                )

                # 用LLaMA推理
                final_answer = ask(full_question)

            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue

            # 保存推理输出
            with open(output_txt_path, "w", encoding="utf-8") as f:
                f.write(final_answer)

            torch.cuda.empty_cache()  # 每处理一个清理显存，防止爆显存

    print("全部LLM推理完成")