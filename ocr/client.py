import base64
import requests
import argparse
import json
import time
import os

def image_to_base64(image_path: str) -> str:
    """将图像转换为base64字符串"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def test_health(server_url: str):
    """测试健康检查端点"""
    try:
        response = requests.get(f"{server_url}/health")
        print(f"Health check status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {str(e)}")
        return False

def test_ocr(server_url: str, image_path: str):
    """测试OCR端点"""
    try:
        # 检查文件是否存在
        if not os.path.exists(image_path):
            print(f"Error: File not found: {image_path}")
            return
            
        # 转换图像为base64
        base64_data = image_to_base64(image_path)
        
        # 发送请求
        start_time = time.time()
        response = requests.post(
            f"{server_url}/ocr",
            json={"image_base64": base64_data}
        )
        processing_time = time.time() - start_time
        
        # 处理响应
        if response.status_code == 200:
            result = response.json()
            print(f"OCR processed in {processing_time:.2f} seconds")
            print(f"Status: {result['status']}")
            
            # 打印识别结果
            if "result" in result and isinstance(result["result"], list):
                for i, item in enumerate(result["result"]):
                    if "rec_texts" in item and item["rec_texts"]:
                        text = item["rec_texts"][0]
                        score = item["rec_scores"][0] if "rec_scores" in item and item["rec_scores"] else 0
                        print(f"Text {i+1}: {text} (confidence: {score:.4f})")
                    else:
                        print(f"Text {i+1}: No text recognized")
            else:
                print("No text recognized in the image")
                    
            # 保存完整结果
            with open("ocr_result.json", "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print("\nFull result saved to ocr_result.json")
        else:
            print(f"Error: {response.status_code}")
            print(f"Response: {response.text}")
    
    except Exception as e:
        print(f"OCR request failed: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PaddleOCR API Client")
    parser.add_argument("--server", default="http://localhost:8000", help="Server URL")
    parser.add_argument("--image", required=True, help="Path to image file")
    args = parser.parse_args()
    
    # 先检查服务健康状态
    if test_health(args.server):
        print("\nService is healthy, sending OCR request...")
        test_ocr(args.server, args.image)