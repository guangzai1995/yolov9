import os
import tempfile
import base64
import uvicorn
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse,PlainTextResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
from kimia_infer.api.kimia import KimiAudio  # 确保已安装

app = FastAPI(title="Kimi-Audio API Service")

# 全局模型实例
model = None
model_loaded = False

DEFAULT_SAMPLING_PARAMS = {
    "audio_temperature": 0.8,
    "audio_top_k": 10,
    "text_temperature": 0.0,
    "text_top_k": 5,
    "audio_repetition_penalty": 1.0,
    "audio_repetition_window_size": 64,
    "text_repetition_penalty": 1.0,
    "text_repetition_window_size": 16,
}


class AudioContent(BaseModel):
    data: str  # base64编码的音频数据
    filename: str  # 原始文件名（可选）
    sample_rate: int = 24000  # 采样率

class Message(BaseModel):
    role: str
    message_type: str
    content: str  # 文本内容或Base64编码的音频数据

class InferenceRequest(BaseModel):
    messages: List[Message]
    output_type: str = "both"
    sampling_params: Optional[Dict] = None

def load_model():
    """启动时加载模型"""
    global model,model_loaded
    try:
        model_id = "/work/moonshotai/Kimi-Audio-7B-Instruct/"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        model = KimiAudio(model_path=model_id, load_detokenizer=True)
        #model.to(device)
        model_loaded = True
        print(f"Model loaded successfully on {device}")
    except Exception as e:
        raise RuntimeError(f"Model loading failed: {str(e)}")
load_model()

def save_base64_audio(base64_data: str) -> str:
    """将Base64音频数据保存为临时文件"""
    try:
        # 解码Base64数据
        audio_bytes = base64.b64decode(base64_data)
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            return tmp.name
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid audio data: {str(e)}")

def has_audio_message(messages: List[Message]) -> bool:
    """检查消息列表中是否包含语音消息"""
    for msg in messages:
        if msg.message_type == "audio":
            return True
    return False


@app.get("/health")
async def health_check():
    """健康检查端点"""
    if model_loaded and model is not None:
        return PlainTextResponse("OK", status_code=200)
    else:
        return PlainTextResponse("Model not loaded", status_code=503)


@app.post("/infer")
async def kimi_inference(request: InferenceRequest):
    """处理推理请求"""
    if model is None or not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not has_audio_message(request.messages):
        raise HTTPException(
            status_code=400,
            detail="Request must contain at least one audio message"
        )

    # 处理临时文件
    temp_files = []
    processed_messages = []
    
    try:
        # 处理消息中的音频数据
        for msg in request.messages:
            if msg.message_type == "audio":
                # 音频消息 - 保存为临时文件
                file_path = save_base64_audio(msg.content)
                temp_files.append(file_path)
                processed_messages.append({
                    "role": msg.role,
                    "message_type": "audio",
                    "content": file_path
                })
            else:
                # 文本消息 - 直接使用
                processed_messages.append(msg.dict())
        
        params = {**DEFAULT_SAMPLING_PARAMS, **(request.sampling_params or {})} 
        
        # 执行推理
        if request.output_type == "text":
            _, text_output = model.generate(processed_messages, **params, output_type="text")
            audio_b64 = None
        else:
            wav_output, text_output = model.generate(processed_messages, **params, output_type="both")
            
            # 将生成的音频保存为Base64
            with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_audio:
                sf.write(tmp_audio.name, wav_output.detach().cpu().view(-1).numpy(), 24000)
                with open(tmp_audio.name, "rb") as f:
                    audio_b64 = base64.b64encode(f.read()).decode("utf-8")
        
        return JSONResponse({
            "text": text_output,
            "audio": audio_b64 if request.output_type != "text" else None,
            "sample_rate": 24000
        })
    
    finally:
        # 清理临时文件
        for f in temp_files:
            if os.path.exists(f):
                os.unlink(f)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)