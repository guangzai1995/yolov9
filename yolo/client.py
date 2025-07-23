import requests

# 服务端地址
SERVER_URL = 'http://localhost:5000/detect'

# 准备请求数据
files = {'image': open('test.jpg', 'rb')}
params = {
    'conf': 0.4,     # 可选参数
    'iou': 0.5,      # 可选参数
    'max_det': 100   # 可选参数
}

# 发送请求
response = requests.post(SERVER_URL, files=files, data=params)

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
        print("获取结果图片失败")
else:
    print("请求失败:", response.text)