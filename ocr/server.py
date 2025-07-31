import base64
import asyncio
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from paddleocr import PaddleOCR
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
        result = ocr.ocr(test_img)
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
    
    # 根据提供的结果，OCR返回的是多层嵌套列表
    # 先展平结果结构
    flattened = []
    
    # 递归展平列表
    def flatten_list(lst):
        for item in lst:
            if isinstance(item, list) and len(item) > 0:
                # 检查是否是我们要找的文本区域结构 [框坐标, (文本, 置信度)]
                if isinstance(item[0], list) and isinstance(item[1], tuple) and len(item[1]) == 2:
                    flattened.append(item)
                else:
                    flatten_list(item)
    
    flatten_list(ocr_result)
    
    # 处理展平后的结果
    for item in flattened:
        try:
            dt_box = item[0]  # 检测框坐标
            rec_text, rec_score = item[1]  # 文本和置信度
            
            # 确保检测框是列表格式
            if hasattr(dt_box, 'tolist'):
                dt_box_list = dt_box.tolist()
            elif isinstance(dt_box, (list, tuple)):
                dt_box_list = [list(coord) if isinstance(coord, (list, tuple, np.ndarray)) else coord 
                              for coord in dt_box]
            else:
                dt_box_list = [dt_box]
            
            # 确保置信度是浮点数
            try:
                rec_score = float(rec_score)
            except (ValueError, TypeError):
                rec_score = 0.0
                
            res_dict = {
                "rec_texts": [rec_text],
                "rec_scores": [rec_score],
                "dt_boxes": dt_box_list
            }
            processed.append(res_dict)
        except Exception as e:
            logger.warning(f"Error processing OCR item: {item}, error: {e}")
    
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
            lambda: ocr.ocr(cv_image)
        )
        
        # 记录原始结果用于调试
        logger.info(f"Raw OCR result type: {type(result)}")
        if isinstance(result, list) and result:
            logger.info(f"First result element type: {type(result[0])}")
            logger.info(f"First result element content: {result[0]}")
        
        # 处理并返回结果
        processed_result = process_ocr_result(result)
        return JSONResponse(content={"status": "success", "result": processed_result})
    
    except Exception as e:
        logger.exception("Error processing OCR request")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Run PaddleOCR API server")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on (default: 8000)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on (default: 0.0.0.0)")
    args = parser.parse_args()
    
    # 使用解析的参数启动服务
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
