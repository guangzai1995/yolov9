import os
import uuid
import base64
import numpy as np
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import cv2
import json
import threading
import concurrent.futures
import torch
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 检查GPU可用性并加载模型到GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = "/work/project/yolov9/model/yolov9e.pt"

# 全局模型加载（确保只加载一次）
model = None
try:
    if device == 'cuda':
        model = YOLO(model_path).to(device)
        logger.info(f"模型已加载到 GPU: {torch.cuda.get_device_name(0)}")
    else:
        model = YOLO(model_path)
        logger.info("模型使用 CPU")
except Exception as e:
    logger.error(f"模型加载失败: {str(e)}")
    model = None

# 配置临时上传目录
base_dir = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(base_dir, 'uploads')
RESULTS_FOLDER = os.path.join(base_dir, 'results')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
logger.info(f"上传目录: {UPLOAD_FOLDER}")
logger.info(f"结果目录: {RESULTS_FOLDER}")

# 创建线程池执行器（异步处理）
executor = concurrent.futures.ThreadPoolExecutor(max_workers=24)

# 任务存储字典（用于异步结果跟踪）
tasks = {}

def process_detection(task_id, upload_path, conf, iou, max_det):
    """异步处理图像检测任务"""
    try:
        # 执行预测
        results = model.predict(
            source=upload_path,
            conf=conf,
            iou=iou,
            max_det=max_det,
            save=False,
            show_labels=True,
            show_conf=True,
            line_width=3,
            device=device  # 确保使用指定设备
        )
        
        # 处理检测结果
        detections = []
        if results[0].boxes is not None:
            boxes = results[0].boxes.cpu().numpy()
            for box in boxes:
                detections.append({
                    "class": int(box.cls[0]),
                    "class_name": results[0].names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "bbox": box.xywhn[0].tolist()
                })
        
        # 生成结果图像
        result_filename = f"result_{task_id}.jpg"
        result_path = os.path.join(RESULTS_FOLDER, result_filename)
        
        for r in results:
            annotated_img = r.plot()
            annotated_img_bgr = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)
            success = cv2.imwrite(result_path, annotated_img_bgr)
            if not success:
                logger.error(f"图片保存失败: {result_path}")
        
        # 更新任务状态为完成
        tasks[task_id] = {
            'status': 'completed',
            'result': {
                'result_image': result_filename,  # 仅返回文件名，不包含路径
                'detections': detections
            }
        }
        logger.info(f"任务 {task_id} 处理完成")
        
    except Exception as e:
        # 更新任务状态为错误
        error_msg = f'处理失败: {str(e)}'
        tasks[task_id] = {
            'status': 'error',
            'message': error_msg
        }
        logger.error(f"任务 {task_id} 出错: {error_msg}")

@app.route('/detect', methods=['POST'])
def detect_objects():
    """异步对象检测端点"""
    # 检查JSON数据中是否包含base64图像
    data = request.get_json()
    if not data or 'image_base64' not in data:
        return jsonify({'error': 'No base64 image provided'}), 400
    
    # 生成唯一任务ID
    task_id = uuid.uuid4().hex
    tasks[task_id] = {'status': 'processing'}
    logger.info(f"新任务提交: {task_id}")
    
    # 获取请求参数（带默认值）
    conf = float(data.get('conf', 0.1))
    iou = float(data.get('iou', 0.1))
    max_det = int(data.get('max_det', 10))
    
    # 解码Base64图像
    try:
        base64_str = data['image_base64']
        if 'base64,' in base64_str:
            base64_str = base64_str.split('base64,')[-1]
        
        img_data = base64.b64decode(base64_str)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 保存解码后的图像
        filename = f"{task_id}.jpg"
        upload_path = os.path.join(UPLOAD_FOLDER, filename)
        cv2.imwrite(upload_path, img)
        logger.info(f"图片保存到: {upload_path}")
        
        # 提交异步任务
        executor.submit(process_detection, task_id, upload_path, conf, iou, max_det)
        
        return jsonify({
            'task_id': task_id,
            'status': 'processing',
            'message': '任务已提交处理'
        })
        
    except Exception as e:
        error_msg = f'无效的图像数据: {str(e)}'
        logger.error(error_msg)
        return jsonify({'error': error_msg}), 400

@app.route('/result/<task_id>', methods=['GET'])
def get_task_result(task_id):
    """获取任务结果"""
    task = tasks.get(task_id)
    
    if not task:
        logger.warning(f"无效的任务ID: {task_id}")
        return jsonify({'error': '无效的任务ID'}), 404
    
    if task['status'] == 'processing':
        return jsonify({
            'task_id': task_id,
            'status': 'processing',
            'message': '任务处理中，请稍后查询'
        })
    
    if task['status'] == 'error':
        return jsonify({
            'task_id': task_id,
            'status': 'error',
            'message': task['message']
        }), 500
    
    return jsonify({
        'task_id': task_id,
        'status': 'completed',
        'result': task['result']
    })

@app.route('/result_image/<filename>')
def get_result_image(filename):
    """返回结果图像"""
    result_path = os.path.join(RESULTS_FOLDER, filename)
    
    # 安全检查：防止路径遍历攻击
    if not os.path.abspath(result_path).startswith(os.path.abspath(RESULTS_FOLDER)):
        logger.warning(f"非法路径访问尝试: {filename}")
        return jsonify({'error': '非法访问'}), 403
    
    if not os.path.exists(result_path):
        logger.error(f"图片不存在: {result_path}")
        return jsonify({'error': '图片不存在'}), 404
    
    logger.info(f"返回图片: {result_path}")
    return send_file(result_path, mimetype='image/jpeg')

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    try:
        # 检查模型是否加载
        if model is None:
            return jsonify({'status': 'error', 'message': '模型未加载'}), 500
        
        # 检查GPU状态
        gpu_status = {
            'available': torch.cuda.is_available(),
            'device_count': torch.cuda.device_count(),
            'current_device': torch.cuda.current_device() if torch.cuda.is_available() else None,
            'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        }
        
        # 简单模型检查（尝试空预测）
        test_input = torch.zeros(1, 3, 640, 640).to(device)
        with torch.no_grad():
            model(test_input)
        
        return jsonify({
            'status': 'healthy',
            'model': 'loaded',
            'device': device,
            'gpu': gpu_status
        })
    
    except Exception as e:
        error_msg = f'健康检查失败: {str(e)}'
        logger.error(error_msg)
        return jsonify({
            'status': 'error',
            'message': error_msg
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)