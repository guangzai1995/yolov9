import os
import uuid
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
    # 检查图片是否上传
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # 保存上传的图片
    filename = secure_filename(file.filename)
    unique_id = uuid.uuid4().hex
    upload_path = os.path.join(UPLOAD_FOLDER, f"{unique_id}_{filename}")
    result_path = os.path.join(RESULTS_FOLDER, f"result_{unique_id}_{filename}")
    file.save(upload_path)
    
    # 获取请求参数（带默认值）
    conf = float(request.form.get('conf', 0.4))
    iou = float(request.form.get('iou', 0.5))
    max_det = int(request.form.get('max_det', 100))
    
    # 执行预测
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
        cv2.imwrite(result_path, im_array[..., ::])  # RGB转BGR
    
    # 返回结果图片路径
    return jsonify({
        'result_image': f"/result/{result_path}",
        'detections': results[0].tojson()
    })

@app.route('/result/<path:filename>')
def get_result_image(filename):
    return send_file(filename, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)