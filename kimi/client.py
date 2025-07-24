# client.py
import requests
import base64
import json

# 基础配置
SERVER_URL = "http://localhost:5000"
HEADERS = {"Content-Type": "application/json"}

def transcribe_audio(audio_path):
    """语音转录 (符合OpenAI API格式)"""
    url = f"{SERVER_URL}/v1/audio/transcriptions"
    
    with open(audio_path, "rb") as f:
        files = {"file": f}
        response = requests.post(url, files=files)
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Transcription failed: {response.text}")

def audio_chat(messages, output_type="text", parameters=None):
    """音频聊天 (自定义格式)"""
    url = f"{SERVER_URL}/v1/audio/chat/completions"
    
    # 处理音频文件
    processed_messages = []
    for msg in messages:
        if msg.get("message_type") == "audio":
            with open(msg["content"], "rb") as f:
                audio_data = base64.b64encode(f.read()).decode("utf-8")
                msg["content"] = f"data:audio/wav;base64,{audio_data}"
        processed_messages.append(msg)
    
    payload = {
        "messages": processed_messages,
        "output_type": output_type,
        "parameters": parameters or {}
    }
    
    response = requests.post(url, headers=HEADERS, data=json.dumps(payload))
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Chat failed: {response.text}")

def save_audio_from_base64(base64_data, output_path):
    """保存base64音频到文件"""
    if base64_data.startswith("data:"):
        base64_data = base64_data.split(",")[1]
    audio_bytes = base64.b64decode(base64_data)
    with open(output_path, "wb") as f:
        f.write(audio_bytes)
    print(f"Audio saved to {output_path}")

# 使用示例
if __name__ == "__main__":
    # 示例1: 语音转录
    asr_result = transcribe_audio("test_audios/asr_example.wav")
    print("Transcription Result:", asr_result["text"])
    
    # 示例2: 纯文本对话
    text_chat_result = audio_chat(
        messages=[
            {"role": "user", "message_type": "text", "content": "你好，请介绍一下你自己"}
        ],
        output_type="text"
    )
    print("Text Response:", text_chat_result["text"])
    
    # 示例3: 带音频输入的对话 (输出音频)
    audio_chat_result = audio_chat(
        messages=[
            {"role": "user", "message_type": "audio", "content": "test_audios/qa_example.wav"}
        ],
        output_type="audio"
    )
    save_audio_from_base64(audio_chat_result["audio"], "response_audio.wav")
    print("Audio response saved")
    
    # 示例4: 混合输入 (文本+音频) 输出两者
    mixed_result = audio_chat(
        messages=[
            {"role": "user", "message_type": "text", "content": "请听以下音频并回答"},
            {"role": "user", "message_type": "audio", "content": "qa_example.wav"}
        ],
        output_type="both"
    )
    print("Text Response:", mixed_result["text"])
    save_audio_from_base64(mixed_result["audio"], "mixed_response.wav")