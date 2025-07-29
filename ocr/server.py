import asyncio
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from paddleocr import PaddleOCR
import numpy as np
import cv2
import base64
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PaddleOCR-Server")

app = FastAPI()
ocr = PaddleOCR(lang='ch')

# 全局模型加载
logger.info("PaddleOCR模型加载完成，服务已就绪")

class ImageRequest(BaseModel):
    image_base64: str
    need_preprocess: bool = False

@app.post("/ocr")
async def process_image(request: ImageRequest):
    try:
        # 解码Base64图像
        logger.info("开始处理OCR请求")
        image_data = base64.b64decode(request.image_base64)
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 可选图像预处理
        if request.need_preprocess:
            logger.info("执行图像预处理")
            img = preprocess_image(img)
        
        # 使用线程池执行OCR
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, ocr.ocr, img)
        
        # 调试：记录原始结果结构
        logger.debug(f"OCR原始结果类型: {type(result)}")
        if result:
            logger.debug(f"第一个元素类型: {type(result[0])}")
            if result[0]:
                logger.debug(f"第一行类型: {type(result[0][0])}")
                logger.debug(f"第一行内容: {result[0][0]}")
        
        # 格式化结果 - 修复解包问题
        formatted = []
        for page in result:  # 处理多页结果
            for line in page:
                # 检查结果结构 - 可能是 [box, (text, confidence)] 或 [box, text, confidence]
                if len(line) == 2:
                    # 标准结构: [坐标, (文本, 置信度)]
                    box, text_conf = line
                    text, confidence = text_conf
                elif len(line) >= 3:
                    # 备用结构: [坐标, 文本, 置信度]
                    box, text, confidence = line[:3]
                else:
                    logger.warning(f"无法识别的行结构: {line}")
                    continue
                
                # 确保box是可序列化的
                if hasattr(box, 'tolist'):
                    box = box.tolist()
                
                formatted.append({
                    "coordinates": box,
                    "text": text,
                    "confidence": float(confidence)
                })
        
        logger.info(f"成功处理OCR请求，识别到 {len(formatted)} 个文本区域")
        return {"success": True, "results": formatted}
    
    except Exception as e:
        logger.error(f"处理OCR请求时出错: {str(e)}")
        return {"success": False, "error": str(e)}

def preprocess_image(img):
    """图像预处理增强"""
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 自适应阈值二值化
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # 可选: 去噪
    denoised = cv2.fastNlMeansDenoising(thresh, h=10)
    
    return denoised

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")