import base64
import asyncio
import numpy as np
from io import BytesIO
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from paddleocr import PaddleOCR
from PIL import Image
import uvicorn
import logging
import cv2

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化OCR模型
def init_ocr():
    logger.info("Initializing PaddleOCR model...")
    ocr = PaddleOCR(
        text_detection_model_name="PP-OCRv5_server_det",
        text_detection_model_dir="/home/ocr_projects/official_models/PP-OCRv5_server_det",
        text_recognition_model_name="PP-OCRv5_server_rec",
        text_recognition_model_dir="/home/ocr_projects/official_models/PP-OCRv5_server_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device='gpu:3'
    )
    logger.info("PaddleOCR model initialized successfully")
    
    # 测试模型是否正常工作
    try:
        logger.info("Running a quick test inference...")
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = ocr.predict(input=test_img)
        logger.info(f"Test inference result type: {type(result)}")
        logger.info(f"Test inference result structure: {result}")
        logger.info("Test inference completed successfully")
    except Exception as e:
        logger.error(f"Test inference failed: {str(e)}")
        raise RuntimeError("PaddleOCR initialization failed") from e
    
    return ocr

# 全局OCR实例
ocr = init_ocr()

app = FastAPI(title="PaddleOCR API", version="1.0")

# 允许所有来源的跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def base64_to_image(base64_str: str) -> np.ndarray:
    """将base64字符串转换为OpenCV图像格式"""
    try:
        # 移除可能的头部信息
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]
            
        img_data = base64.b64decode(base64_str)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image data")
        return img
    except Exception as e:
        logger.error(f"Error decoding base64 image: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

def process_ocr_result(ocr_result):
    """处理OCR结果，转换为可序列化的格式"""
    processed = []
    
    # 检查结果结构 - 可能是列表或字典
    if isinstance(ocr_result, list) and len(ocr_result) > 0:
        # 新版本PaddleOCR返回列表，每个元素是一个字典
        for res in ocr_result:
            # 获取识别文本和置信度
            rec_texts = res.get('rec_texts', [])
            rec_scores = res.get('rec_scores', [])
            
            # 如果rec_texts为空，尝试从其他字段获取
            if not rec_texts:
                rec_res = res.get('rec_res', [])
                if rec_res:
                    rec_texts = [item[0] for item in rec_res]
                    rec_scores = [float(item[1]) for item in rec_res]
            
            # 获取检测框
            dt_boxes = res.get('dt_boxes', [])
            dt_polys = res.get('dt_polys', [])
            
            # 如果dt_boxes为空，尝试从其他字段获取
            if not dt_boxes and 'boxes' in res:
                dt_boxes = res['boxes']
            
            # 转换结果格式
            res_dict = {
                "rec_texts": rec_texts,
                "rec_scores": rec_scores,
                "dt_boxes": [box.tolist() if hasattr(box, 'tolist') else box for box in dt_boxes],
                "dt_polys": [poly.tolist() if hasattr(poly, 'tolist') else poly for poly in dt_polys],
            }
            
            # 添加其他可能存在的字段
            for key in ['input_path', 'page_index', 'textline_orientation_angles']:
                if key in res:
                    value = res[key]
                    if hasattr(value, 'tolist'):
                        value = value.tolist()
                    res_dict[key] = value
            
            processed.append(res_dict)
    
    else:
        # 旧版本PaddleOCR可能返回对象
        logger.warning("Unexpected OCR result format. Attempting to process as object.")
        try:
            # 尝试使用旧版属性访问
            res_dict = {
                "rec_texts": ocr_result.rec_texts,
                "rec_scores": [float(score) for score in ocr_result.rec_scores],
                "dt_boxes": [box.tolist() for box in ocr_result.dt_boxes],
                "dt_polys": [poly.tolist() for poly in ocr_result.dt_polys],
            }
            
            # 添加其他可能存在的字段
            if hasattr(ocr_result, 'textline_orientation_angles'):
                res_dict["textline_orientation_angles"] = ocr_result.textline_orientation_angles.tolist()
            
            if hasattr(ocr_result, 'input_path'):
                res_dict["input_path"] = ocr_result.input_path
                
            processed.append(res_dict)
        except Exception as e:
            logger.error(f"Failed to process OCR result: {str(e)}")
            logger.error(f"Result structure: {ocr_result}")
    
    return processed

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy", "model": "PP-OCRv5_server"}

@app.post("/ocr")
async def ocr_endpoint(image_data: dict):
    """OCR处理端点"""
    try:
        base64_str = image_data.get("image_base64")
        if not base64_str:
            raise HTTPException(status_code=400, detail="Missing image_base64 field")
        
        # 转换base64为图像
        cv_image = base64_to_image(base64_str)
        
        # 在后台线程中运行OCR（避免阻塞事件循环）
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, 
            lambda: ocr.predict(input=cv_image)
        )
        
        # 记录原始结果用于调试
        logger.info(f"Raw OCR result type: {type(result)}")
        if isinstance(result, list) and result:
            logger.info(f"First result element type: {type(result[0])}")
        
        # 处理并返回结果
        processed_result = process_ocr_result(result)
        return JSONResponse(content={"status": "success", "result": processed_result})
    
    except Exception as e:
        logger.exception("Error processing OCR request")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")