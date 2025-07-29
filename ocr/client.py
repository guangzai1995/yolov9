import asyncio
import aiohttp
import sys
import base64
import json
import time
import os

async def send_image(image_path, need_preprocess=False):
    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"错误: 文件 '{image_path}' 不存在")
        return {"success": False, "error": "文件不存在"}
    
    # 读取并编码图像
    try:
        with open(image_path, 'rb') as f:
            image_data = f.read()
            image_base64 = base64.b64encode(image_data).decode('utf-8')
    except Exception as e:
        print(f"读取文件出错: {str(e)}")
        return {"success": False, "error": str(e)}
    
    # 准备请求数据
    payload = {
        "image_base64": image_base64,
        "need_preprocess": need_preprocess
    }
    
    # 发送请求
    start_time = time.time()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                'http://localhost:8000/ocr',
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=60  # 60秒超时
            ) as response:
                result = await response.json()
                elapsed = time.time() - start_time
                result['processing_time'] = f"{elapsed:.2f}秒"
                return result
    except Exception as e:
        print(f"请求服务器出错: {str(e)}")
        return {"success": False, "error": str(e)}

def print_result(result):
    if not result.get('success'):
        print(f"OCR失败: {result.get('error', '未知错误')}")
        return
    
    print("\nOCR结果:")
    for idx, item in enumerate(result['results'], 1):
        print(f"{idx}. 文本: {item['text']}")
        print(f"   置信度: {item['confidence']:.4f}")
        print(f"   坐标: {item['coordinates']}")
    
    # 统计信息
    total_items = len(result['results'])
    total_chars = sum(len(item['text']) for item in result['results'])
    avg_confidence = sum(item['confidence'] for item in result['results']) / total_items if total_items > 0 else 0
    
    print("\n统计:")
    print(f" - 识别区域数: {total_items}")
    print(f" - 总字符数: {total_chars}")
    print(f" - 平均置信度: {avg_confidence:.4f}")
    print(f" - 处理时间: {result.get('processing_time', '未知')}")

def main():
    
    image_path = "test.jpg"
    need_preprocess = False
    
    print(f"处理图像: {image_path}")
    if need_preprocess:
        print("启用图像预处理")
    
    try:
        result = asyncio.run(send_image(image_path, need_preprocess))
        print_result(result)
    except Exception as e:
        print(f"客户端运行时错误: {str(e)}")

if __name__ == "__main__":
    main()