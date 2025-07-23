# import paddlex as pdx
# from paddlex.inference import download_model

# # 下载文档方向分类模型
# download_model("PP-LCNet_x1_0_doc_ori", task_type="doc_orientation")

from paddlex import create_model
 
# 1. 创建模型实例
model = create_model(model_name="UVDoc")
 