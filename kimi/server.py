# server.py
import os
import time
import uuid
import base64
import soundfile as sf
import torch
from flask import Flask, request, jsonify
from kimia_infer.api.kimia import KimiAudio

app = Flask(__name__)

# 全局模型实例
model = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def initialize_model():
    global model
    model_id = "/work/model/moonshotai/Kimi-Audio-7B-Instruct/"
    try:
        model = KimiAudio(model_path=model_id, load_detokenizer=True)
        model.to(device)
        print("Model loaded successfully on device:", device)
    except Exception as e:
        print(f"Model loading failed: {e}")
        raise RuntimeError("Model initialization failed")

# 默认采样参数
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

@app.before_first_request
def before_first_request():
    initialize_model()

def save_audio_file(audio_data, filename):
    """保存上传的音频文件并返回路径"""
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, filename)
    
    try:
        # 尝试解码base64
        if isinstance(audio_data, str) and audio_data.startswith("data:"):
            header, data = audio_data.split(",", 1)
            audio_bytes = base64.b64decode(data)
            with open(file_path, "wb") as f:
                f.write(audio_bytes)
        else:
            # 直接保存二进制数据
            with open(file_path, "wb") as f:
                f.write(audio_data)
        return file_path
    except Exception as e:
        print(f"Error saving audio: {e}")
        return None

@app.route('/v1/audio/transcriptions', methods=['POST'])
def transcribe_audio():
    """语音转录端点 (符合OpenAI格式)"""
    start_time = time.time()
    
    if 'file' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    audio_file = request.files['file']
    filename = f"transcribe_{uuid.uuid4()}.wav"
    audio_path = save_audio_file(audio_file.read(), filename)
    
    if not audio_path:
        return jsonify({"error": "Failed to process audio file"}), 500

    try:
        # 构建消息
        messages = [
            {"role": "user", "message_type": "text", "content": "请转录以下音频内容："},
            {"role": "user", "message_type": "audio", "content": audio_path}
        ]
        
        # 获取请求参数或使用默认值
        params = request.form.to_dict()
        sampling_params = {**DEFAULT_SAMPLING_PARAMS, **params}
        
        # 执行转录
        _, text_output = model.generate(messages, **sampling_params, output_type="text")
        
        # 清理临时文件
        if os.path.exists(audio_path):
            os.remove(audio_path)
        
        # 构建OpenAI格式响应
        return jsonify({
            "text": text_output,
            "processing_time": round(time.time() - start_time, 2)
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/v1/audio/chat/completions', methods=['POST'])
def audio_chat():
    """音频聊天端点 (自定义扩展格式)"""
    start_time = time.time()
    data = request.json
    
    if not data or "messages" not in data:
        return jsonify({"error": "Invalid request format"}), 400
    
    # 处理消息中的音频
    processed_messages = []
    for msg in data["messages"]:
        if msg.get("message_type") == "audio" and "content" in msg:
            filename = f"chat_{uuid.uuid4()}.wav"
            audio_path = save_audio_file(msg["content"], filename)
            if audio_path:
                msg["content"] = audio_path
        processed_messages.append(msg)
    
    # 获取输出类型和参数
    output_type = data.get("output_type", "text")
    params = data.get("parameters", {})
    sampling_params = {**DEFAULT_SAMPLING_PARAMS, **params}
    
    try:
        # 执行生成
        if output_type == "text":
            _, text_output = model.generate(processed_messages, **sampling_params, output_type="text")
            audio_output = None
        elif output_type == "audio":
            wav_output, _ = model.generate(processed_messages, **sampling_params, output_type="audio")
            audio_output = wav_output.detach().cpu().view(-1).numpy()
            text_output = None
        else:  # both
            wav_output, text_output = model.generate(processed_messages, **sampling_params, output_type="both")
            audio_output = wav_output.detach().cpu().view(-1).numpy()
        
        # 处理音频输出
        audio_data = None
        if audio_output is not None:
            output_path = f"output_{uuid.uuid4()}.wav"
            sf.write(output_path, audio_output, 24000)
            with open(output_path, "rb") as f:
                audio_data = base64.b64encode(f.read()).decode("utf-8")
            os.remove(output_path)
        
        # 清理输入音频文件
        for msg in processed_messages:
            if isinstance(msg.get("content"), str) and msg.get("message_type") == "audio":
                if os.path.exists(msg["content"]):
                    os.remove(msg["content"])
        
        # 构建响应
        response = {
            "text": text_output,
            "audio": audio_data,
            "output_type": output_type,
            "processing_time": round(time.time() - start_time, 2)
        }
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)