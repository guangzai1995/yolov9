import os
import uuid
import base64  # 新增base64模块
import numpy as np  # 新增numpy模块
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import cv2

app = Flask(__name__)
model = YOLO("/work/project/yolov9/model/yolov9e.pt")

# 配置临时上传目录
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

@app.route('/detect', methods=['POST'])
def detect_objects():
    # 检查JSON数据中是否包含base64图像
    data = request.get_json()
    if not data or 'image_base64' not in data:
        return jsonify({'error': 'No base64 image provided'}), 400

    # 生成唯一文件名
    unique_id = uuid.uuid4().hex
    filename = f"{unique_id}.jpg"
    upload_path = os.path.join(UPLOAD_FOLDER, filename)
    result_path = os.path.join(RESULTS_FOLDER, f"result_{filename}")

    try:
        # 解码Base64图像（移除可能的头部信息）
        base64_str = data['image_base64']
        if 'base64,' in base64_str:
            base64_str = base64_str.split('base64,')[-1]
        
        img_data = base64.b64decode(base64_str)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 保存解码后的图像
        cv2.imwrite(upload_path, img)
    except Exception as e:
        return jsonify({'error': f'Invalid image data: {str(e)}'}), 400

    # 获取请求参数（带默认值）
    conf = float(data.get('conf', 0.4))
    iou = float(data.get('iou', 0.5))
    max_det = int(data.get('max_det', 100))
    
    # 执行预测
    try:
        results = model.predict(
            source=upload_path,
            conf=conf,
            iou=iou,
            max_det=max_det,
            save=True,
            project=RESULTS_FOLDER,
            name='',
            exist_ok=True,
            show_labels=True,
            show_conf=True,
            line_width=3
        )
        
        # 保存结果图片
        for r in results:
            im_array = r.plot()
            cv2.imwrite(result_path, im_array[..., ::-1])  # 注意颜色通道转换
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
    
    # 返回结果图片路径
    return jsonify({
        'result_image': f"/result/result_{filename}",
        'detections': results[0].tojson()
    })

@app.route('/result/<path:filename>')
def get_result_image(filename):
    result_path = os.path.join(RESULTS_FOLDER, filename)
    return send_file(result_path, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)