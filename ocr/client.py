import argparse
import base64
import json
import logging
import os
import requests
from typing import Dict, Any, List

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class OCRClient:
    """OCR服务客户端"""
    
    def __init__(self, api_url: str):
        """初始化OCR客户端"""
        self.api_url = api_url
        
    def process_file(self, file_path: str, output_dir: str = ".") -> None:
        """处理单个文件并保存OCR结果"""
        try:
            # 检查文件是否存在
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"文件不存在: {file_path}")
                
            # 读取并编码文件
            with open(file_path, "rb") as file:
                file_bytes = file.read()
                file_data = base64.b64encode(file_bytes).decode("ascii")
                
            # 准备请求
            payload = {"file": file_data, "fileType": 1}
            
            # 发送请求
            logger.info(f"正在发送请求到 {self.api_url}")
            response = requests.post(self.api_url, json=payload)
            
            # 检查响应状态
            response.raise_for_status()
            
            # 处理响应
            result = response.json()
            
            # 验证响应结构
            if "result" not in result or "ocrResults" not in result["result"]:
                raise ValueError("无效的响应格式")
                
            # 创建输出目录（如果不存在）
            os.makedirs(output_dir, exist_ok=True)
            
            # 处理OCR结果
            ocr_results = result["result"]["ocrResults"]
            logger.info(f"成功获取 {len(ocr_results)} 个OCR结果")
            
            for i, res in enumerate(ocr_results):
                # 提取文本结果
                text_result = res.get("prunedResult", "")
                print(f"结果 {i+1}: {text_result}")
                
                # 保存OCR图像（如果存在）
                ocr_image_data = res.get("ocrImage")
                if ocr_image_data:
                    ocr_img_path = os.path.join(output_dir, f"ocr_{i}.jpg")
                    with open(ocr_img_path, "wb") as f:
                        f.write(base64.b64decode(ocr_image_data))
                    logger.info(f"图像已保存至 {ocr_img_path}")
                    
        except Exception as e:
            logger.error(f"处理文件时出错: {str(e)}")
            raise

def main():
    """主函数"""
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="OCR服务客户端")
    parser.add_argument("--api-url", default="http://localhost:8206/ocr", help="OCR API的URL")
    parser.add_argument("--input-file", required=True, help="要处理的文件路径")
    parser.add_argument("--output-dir", default=".", help="输出结果的目录")
    parser.add_argument("--verbose", action="store_true", help="启用详细日志")
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    try:
        # 创建OCR客户端并处理文件
        client = OCRClient(args.api_url)
        client.process_file(args.input_file, args.output_dir)
        
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()    