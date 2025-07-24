import requests
import base64

# 服务端地址
SERVER_URL = 'http://localhost:5000/detect'

# 读取图像并转换为Base64
with open('test.jpg', 'rb') as image_file:
    # 直接编码为Base64字符串
    base64_image = base64.b64encode(image_file.read()).decode('utf-8')

# 准备请求数据 - 现在使用JSON格式
payload = {
    'image_base64': base64_image,  # 必须的Base64图像数据
    'conf': 0.4,                   # 可选参数
    'iou': 0.5,                    # 可选参数
    'max_det': 100                 # 可选参数
}

# 发送请求 - 现在使用JSON格式
headers = {'Content-Type': 'application/json'}
response = requests.post(SERVER_URL, json=payload, headers=headers)

if response.status_code == 200:
    result = response.json()
    print("检测结果:", result['detections'])
    
    # 下载结果图片
    result_image_url = f"http://localhost:5000{result['result_image']}"
    img_response = requests.get(result_image_url)
    
    if img_response.status_code == 200:
        with open('result_image.jpg', 'wb') as f:
            f.write(img_response.content)
        print("结果图片已保存至: result_image.jpg")
    else:
        print("获取结果图片失败, 状态码:", img_response.status_code)
else:
    print("请求失败, 状态码:", response.status_code)
    print("错误信息:", response.text)