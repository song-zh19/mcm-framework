import json


with open("/DATA/DATANAS1/hrz/azhospital/SeqCls/iKang/ecg_data_processed/xianweihua/genEmbedding-qwen2_5/textinfo.json", "r") as f:
    info = json.load(f)


sentences = info["sentence"]

import re

def process_item(item):
    # 提取所有数字（支持整数和浮点数，包括科学计数法）
    if "超声检查所见" in item:
        return None
    try:
        idx = item.index('为')
        item = item[idx:]
        if '*10' in item:
            idx2 = item.index('*10')
            item = item[:idx2]
    except:
        return None
    numbers = re.findall(r'[+-]?(\d+\.?\d*[eE]?\d*|\.\d+[eE]?\d*)', item)
    if not numbers:
        return None
    # 转换为浮点数并计算平均值
    try:
        float_numbers = [float(n) for n in numbers]
        return sum(float_numbers) / len(float_numbers) if len(float_numbers) > 1 else float_numbers[0]
    except:
        return None  # 处理无法转换为数字的异常情况

# 处理后的结果列表
clinical_value = {}
for key, value in sentences.items():
    clinical_value[key] = [process_item(item) for item in value]

with open("/DATA/DATANAS1/hrz/azhospital/SeqCls/iKang/ecg_data_processed/xianweihua/genEmbedding-qwen2_5/clinical_value2.json", "w") as f:
    json.dump(clinical_value, f)
