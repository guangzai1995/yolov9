import requests
import base64
import json
from pydantic import BaseModel
from typing import List, Dict, Optional

# 定义与服务器相同的请求结构
class Message(BaseModel):
    role: str
    message_type: str
    content: str  # 文本内容或Base64编码的音频数据

class InferenceRequest(BaseModel):
    messages: List[Message]
    output_type: str = "both"
    sampling_params: Optional[Dict] = None

def audio_to_base64(file_path: str) -> str:
    """将音频文件转换为Base64字符串"""
    with open(file_path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode('utf-8')

def send_audio_request():
    """发送请求到Kimi-Audio服务"""
    url = "http://localhost:5000/infer"
    
    # 读取音频文件并转换为Base64
    #audio_base64 = audio_to_base64("test_audios/asr_example.wav")
    audio_base64 = audio_to_base64("test_audios/qa_example.wav")
    
    # 构造请求消息
    messages = [
        #{"role": "user", "message_type": "text", "content": "你好！"},
        {"role": "user", "message_type": "audio", "content": audio_base64}
    ]
    
    # 创建请求体
    request_data = InferenceRequest(
        messages=messages,
        output_type="both",
        #output_type="text",
        sampling_params={
            "audio_temperature": 0.95,  # 覆盖默认值
            "audio_top_k": 15,          # 覆盖默认值
        }
    )
    
    # 发送请求
    headers = {"Content-Type": "application/json"}
    response = requests.post(
        url,
        data=request_data.model_dump_json(),
        headers=headers
    )
    
    if response.status_code == 200:
        result = response.json()
        print("文本输出:", result["text"])
        
        if result.get("audio"):
            # 保存音频文件
            with open("response_audio.wav", "wb") as f:
                f.write(base64.b64decode(result["audio"]))
            print("音频已保存到: response_audio.wav")
    else:
        print(f"请求失败: {response.status_code}")
        print(response.text)

def check_health():
    """检查服务健康状态"""
    url = "http://localhost:5000/health"
    response = requests.get(url)
    
    if response.status_code == 200:
        print("服务状态: 健康 (200 OK)")
    else:
        print(f"服务状态: 异常 ({response.status_code} - {response.text})")


if __name__ == "__main__":
    check_health()
    send_audio_request()