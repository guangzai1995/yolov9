import asyncio
import aiohttp
import base64
import json
from pathlib import Path
from typing import Optional

class OCRClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def image_to_base64(self, image_path: str) -> str:
        """将图片文件转换为base64编码"""
        try:
            with open(image_path, "rb") as image_file:
                encoded = base64.b64encode(image_file.read()).decode('utf-8')
                return encoded
        except Exception as e:
            raise ValueError(f"图片编码失败: {e}")
    
    async def check_health(self) -> dict:
        """检查服务健康状态"""
        if not self.session:
            raise RuntimeError("客户端未初始化")
        
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {
                        "status": "error",
                        "message": f"HTTP {response.status}: {await response.text()}"
                    }
        except Exception as e:
            return {
                "status": "error",
                "message": f"连接失败: {e}"
            }
    
    async def ocr_predict(self, image_path: str, lang: str = "ch") -> dict:
        """执行OCR识别"""
        if not self.session:
            raise RuntimeError("客户端未初始化")
        
        try:
            # 检查文件大小
            file_size = Path(image_path).stat().st_size
            print(f"图片文件大小: {file_size / 1024:.2f} KB")
            
            # 转换图片为base64
            image_base64 = self.image_to_base64(image_path)
            print(f"Base64编码长度: {len(image_base64)}")
            
            # 准备请求数据
            data = {
                "image_base64": image_base64,
                "lang": lang
            }
            
            # 发送请求
            async with self.session.post(
                f"{self.base_url}/ocr",
                json=data,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=60)  # 增加超时时间
            ) as response:
                
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    print(f"服务器错误响应: {error_text}")  # 调试信息
                    return {
                        "success": False,
                        "message": f"HTTP {response.status}: {error_text}",
                        "results": []
                    }
                    
        except Exception as e:
            print(f"客户端请求异常: {e}")  # 调试信息
            return {
                "success": False,
                "message": f"请求失败: {e}",
                "results": []
            }

# 使用示例
async def main():
    async with OCRClient() as client:
        # 检查服务状态
        print("检查服务健康状态...")
        health = await client.check_health()
        print(f"健康状态: {health}")
        
        if health.get("status") != "healthy":
            print("服务不可用，退出")
            return
        
        # 执行OCR识别
        image_path = "test.jpg"  # 请确保图片存在
        if Path(image_path).exists():
            print(f"\n开始识别图片: {image_path}")
            result = await client.ocr_predict(image_path)
            
            if result["success"]:
                print(f"识别成功！识别结果: {result['results']}")
            else:
                print(f"识别失败: {result['message']}")
        else:
            print(f"图片文件不存在: {image_path}")

if __name__ == "__main__":
    asyncio.run(main())