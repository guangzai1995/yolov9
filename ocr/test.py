from paddleocr import PaddleOCR
ocr = PaddleOCR(lang='zh') # need to run only once to load model into memory
img_path = "/work/project/paddleOcr/论文1.jpg"
result = ocr.ocr(img_path)
for idx in range(len(result)):
    res = result[idx]
    for line in res:
        print(line)
