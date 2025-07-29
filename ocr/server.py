from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
import io
import asyncio
from concurrent.futures import ThreadPoolExecutor
from paddleocr import PaddleOCR
import numpy as np
from PIL import Image
import uvicorn

app = FastAPI(title="OCR服务", description="基于PaddleOCR的文字识别服务")

# 全局OCR实例
ocr = None
executor = ThreadPoolExecutor(max_workers=4)

class OCRRequest(BaseModel):
    image_base64: str
    lang: str = "ch"

class OCRResponse(BaseModel):
    success: bool
    message: str
    results: list = []

class HealthResponse(BaseModel):
    status: str
    message: str

@app.on_event("startup")
async def startup_event():
    """应用启动时初始化OCR模型"""
    global ocr
    try:
        ocr = PaddleOCR(lang='ch')
        print("OCR模型加载成功")
    except Exception as e:
        print(f"OCR模型加载失败: {e}")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查接口"""
    if ocr is None:
        return HealthResponse(
            status="unhealthy",
            message="OCR模型未加载"
        )
    return HealthResponse(
        status="healthy",
        message="服务运行正常"
    )

def decode_base64_image(base64_str: str) -> np.ndarray:
    """解码base64图片为numpy数组"""
    try:
        # 移除base64前缀（如果有的话）
        if base64_str.startswith('data:image'):
            base64_str = base64_str.split(',')[1]
        
        # 解码base64
        image_data = base64.b64decode(base64_str)
        
        # 转换为PIL图片
        image = Image.open(io.BytesIO(image_data))
        
        # 转换为RGB格式（如果需要）
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 转换为numpy数组
        image_array = np.array(image)
        
        return image_array
    except Exception as e:
        raise ValueError(f"图片解码失败: {e}")

def run_ocr(image_array: np.ndarray) -> list:
    """在线程池中运行OCR识别"""
    try:
        result = ocr.predict(image_array)
        print(f"OCR原始结果: {result}")  # 调试日志
        return result
    except Exception as e:
        raise RuntimeError(f"OCR识别失败: {e}")

@app.post("/ocr", response_model=OCRResponse)
async def ocr_predict(request: OCRRequest):
    """OCR文字识别接口"""
    if ocr is None:
        raise HTTPException(status_code=503, detail="OCR服务未初始化")
    
    try:
        # 解码base64图片
        image_array = decode_base64_image(request.image_base64)
        
        # 异步运行OCR识别
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, run_ocr, image_array)
        
        # 处理识别结果
        formatted_results = []
        
        # PaddleOCR结果格式检查和处理
        if result is not None and len(result) > 0:
            # result是一个列表，通常只有一个元素（对应一张图片）
            page_result = result[0] if result[0] is not None else []
            
            for line_info in page_result:
                try:
                    if line_info and len(line_info) >= 2:
                        # line_info格式: [bbox, (text, confidence)]
                        bbox = line_info[0] if line_info[0] else []
                        text_info = line_info[1] if line_info[1] else ("", 0.0)
                        
                        # 处理文本信息
                        if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                            text = str(text_info[0]) if text_info[0] else ""
                            confidence = float(text_info[1]) if text_info[1] else 0.0
                        else:
                            text = str(text_info) if text_info else ""
                            confidence = 0.0
                        
                        formatted_results.append({
                            "text": text,
                            "confidence": confidence,
                            "bbox": bbox
                        })
                        
                except Exception as e:
                    print(f"处理单行结果时出错: {e}, line_info: {line_info}")
                    continue
        
        return OCRResponse(
            success=True,
            message=f"识别成功，共识别到{len(formatted_results)}行文字",
            results=formatted_results
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        print(f"服务器内部错误详情: {e}")  # 调试日志
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {e}")

@app.get("/")
async def root():
    """根路径"""
    return {"message": "OCR服务运行中", "docs": "/docs"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")