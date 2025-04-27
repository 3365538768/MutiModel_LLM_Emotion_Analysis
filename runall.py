import subprocess
import time

def run_script(script_name):
    print(f"开始运行 {script_name} ...")
    result = subprocess.run(["python", script_name])
    if result.returncode != 0:
        print(f"运行 {script_name} 失败，终止程序！")
        exit(1)
    print(f"{script_name} 运行完成 ✅")
    print("-" * 50)
    time.sleep(5)  # 每个脚本之间停5秒，防止资源冲突

if __name__ == "__main__":
    script_list = [
        "audio_emotion_analyse.py",
        "preprocess.py",
        "demo_v2.py",
        "LLM_final_emotion_analyse.py",
        "train_emotion_classification.py"
    ]

    for script in script_list:
        run_script(script)

    print("全部脚本运行完成！🎉")
