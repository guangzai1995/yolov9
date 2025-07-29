import requests
import base64
import time
import os

# 服务端地址
SERVER_URL = 'http://localhost:5000/detect'
RESULT_URL = 'http://localhost:5000/result/'
RESULT_IMAGE_URL = 'http://localhost:5000/result_image/'
HEALTH_URL = 'http://localhost:5000/health'

# 健康检查
print("执行健康检查...")
try:
    health_response = requests.get(HEALTH_URL, timeout=10)
    if health_response.status_code == 200:
        health_data = health_response.json()
        print("健康检查结果:")
        print(f"  状态: {health_data['status']}")
        print(f"  设备: {health_data['device']}")
        if health_data['gpu']['available']:
            print(f"  GPU: {health_data['gpu']['device_name']}")
        else:
            print("  GPU: 不可用")
    else:
        print(f"健康检查失败, 状态码: {health_response.status_code}")
        print(health_response.text)
except Exception as e:
    print(f"健康检查错误: {str(e)}")
    exit()

# 读取图像
image_path = 'test.jpg'
if not os.path.exists(image_path):
    print(f"错误: 测试图片不存在: {image_path}")
    exit()

print(f"\n读取测试图片: {image_path}")
with open(image_path, 'rb') as image_file:
    # 直接编码为Base64字符串
    base64_image = base64.b64encode(image_file.read()).decode('utf-8')

# 准备请求数据
print("\n提交检测任务...")
payload = {
    'image_base64': base64_image,
    'conf': 0.25,
    'iou': 0.45,
    'max_det': 100
}

try:
    # 发送请求
    response = requests.post(SERVER_URL, json=payload, timeout=30)
    
    if response.status_code != 200:
        print(f"提交任务失败, 状态码: {response.status_code}")
        print(response.text)
        exit()
    
    task_data = response.json()
    task_id = task_data['task_id']
    print(f"任务已提交, ID: {task_id}, 状态: {task_data['status']}")
    
    # 轮询任务结果
    print("\n等待处理结果...")
    max_retries = 10
    retry_count = 0
    wait_time = 3  # 初始等待时间
    
    while retry_count < max_retries:
        try:
            result_response = requests.get(f"{RESULT_URL}{task_id}", timeout=10)
            result_data = result_response.json()
            
            if result_data['status'] == 'completed':
                print("\n检测完成!")
                print("检测结果:")
                for detection in result_data['result']['detections']:
                    print(f"- {detection['class_name']}: 置信度 {detection['confidence']:.2f}, "
                          f"位置 {detection['bbox']}")
                
                # 下载结果图片
                image_filename = result_data['result']['result_image']
                img_url = f"{RESULT_IMAGE_URL}{image_filename}"
                print(f"\n下载结果图片: {img_url}")
                
                img_response = requests.get(img_url, timeout=30)
                
                if img_response.status_code == 200:
                    output_path = f"result_{task_id}.jpg"
                    with open(output_path, 'wb') as f:
                        f.write(img_response.content)
                    print(f"结果图片已保存至: {output_path}")
                else:
                    print(f"获取结果图片失败, 状态码: {img_response.status_code}")
                break
                
            elif result_data['status'] == 'error':
                print("\n处理错误:", result_data['message'])
                break
                
            else:
                print(f"任务处理中... (等待{wait_time}秒后重试)")
                time.sleep(wait_time)
                retry_count += 1
                # 指数退避策略
                wait_time = min(wait_time * 1.5, 30)
                
        except requests.exceptions.RequestException as e:
            print(f"请求错误: {str(e)}")
            retry_count += 1
            time.sleep(wait_time)
            wait_time = min(wait_time * 1.5, 30)
    
    if retry_count >= max_retries:
        print("\n错误: 超过最大重试次数，任务可能仍在处理中或失败")
        
except requests.exceptions.RequestException as e:
    print(f"网络错误: {str(e)}")
except Exception as e:
    print(f"未知错误: {str(e)}")