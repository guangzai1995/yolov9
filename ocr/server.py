from flask import Flask, request, jsonify
from paddleocr import PaddleOCR
import base64
import numpy as np
import cv2
import logging
import traceback

app = Flask(__name__)

# 全局初始化OCR模型
ocr_engine = PaddleOCR(
    use_textline_orientation=True, 
    text_recognition_batch_size=16,
    device='cpu',  # 使用GPU加速
    # use_angle_cls=True,  # 方向分类在初始化时设置，不在调用时设置
    lang='ch',
    # show_log=False
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_image(image_b64):
    """处理Base64编码图像并返回OpenCV格式"""
    try:
        img_bytes = base64.b64decode(image_b64)
        img_np = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image data")
        return img
    except Exception as e:
        logger.error(f"Image processing failed: {str(e)}")
        raise

@app.route('/ocr', methods=['POST'])
def ocr_api():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        image_b64 = data.get('image')
        if not image_b64:
            return jsonify({"error": "No image provided"}), 400

        # 图像处理
        img = process_image(image_b64)
        
        
        result = ocr_engine.predict(img)
        
        # 结果解析
        text_list = []
        if result and result[0]:
            for line in result[0]:
                if len(line) >= 2 and isinstance(line[1], (list, tuple)):
                    text = line[1][0]
                    text_list.append(text)
        
        return jsonify({"text": text_list})
        
    except Exception as e:
        logger.error(f"OCR processing error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    # 打印PaddleOCR版本信息以便调试
    from paddleocr import __version__ as ocr_version
    logger.info(f"Starting OCR server with PaddleOCR v{ocr_version}")
    
    app.run(host='0.0.0.0', port=5000, threaded=True)