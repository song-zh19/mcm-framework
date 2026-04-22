import re
import os
import json

import numpy as np
import pandas as pd

from tqdm import tqdm
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer


# with open("/home/user/code/stat/logs/textDict.json", "r") as f:
#     textDict = json.load(f)
# with open("/home/user/code/stat/logs/sentencesDict.json", "r") as f:
#     sentencesDict = json.load(f)
# outputPath = "/home/user/code/SeqCls/ecg_data_processed/fangchan_fs100/qwen2Embedding"
# if not os.path.exists(outputPath):
#     os.makedirs(outputPath)

# with open("/home/user/code/stat/logs/workno2filename.json", "r") as f:
#     workno2filename = json.load(f)


# device = "cuda" # the device to load the model onto
# model = AutoModelForCausalLM.from_pretrained(
#     "/home/user/code/SeqCls/checkpoints/Qwen2.5-7B-Instruct/Qwen2.5-7B-Instruct",
#     torch_dtype="auto",
#     device_map="auto"
# )
# tokenizer = AutoTokenizer.from_pretrained("/home/user/code/SeqCls/checkpoints/Qwen2.5-7B-Instruct/Qwen2.5-7B-Instruct")

# responseDict = {}

# for workno, filename in tqdm(workno2filename.items()):
#     if workno not in textDict:
#         print(workno)
#         continue
#     text = textDict[workno]
#     sentences = sentencesDict[workno]
#     if len(text) > 8000:
#         text = text[:8000]

#     messages = [
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": '假设你是经验丰富的医生，基于以下医学报告，对该患者的健康状况进行详细的分析，推测该患者潜在的可能疾病，撰写一份详细的报告（不得摘抄原文本中的结果，必须给出你自己的见解）:' + text}
#     ]
#     text = tokenizer.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=True
#     )
#     model_inputs = tokenizer([text], return_tensors="pt").to(device)

#     generated_ids = model.generate(**model_inputs, max_new_tokens=1024)
#     generated_ids = [
#         output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
#     ]

#     response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
#     responseDict[workno] = response
#     with open(f"/home/user/code/stat/logs/responseDict/{workno}.txt", "w") as f:
#         f.write(response)

# with open("/home/user/code/stat/logs/responseDict.json", "w") as f:
#     json.dump(responseDict, f)


with open("/home/user/code/stat/logs/textDict.json", "r") as f:
    textDict = json.load(f)
with open("/home/user/code/stat/logs/sentencesDict.json", "r") as f:
    sentencesDict = json.load(f)
with open("/home/user/code/stat/logs/responseDict.json", "r") as f:
    responseDict = json.load(f)
outputPath = "/home/user/code/SeqCls/ecg_data_processed/fangchan_fs100/qwen2Embedding_with_LLM"
if not os.path.exists(outputPath):
    os.makedirs(outputPath)

with open("/home/user/code/stat/logs/workno2filename.json", "r") as f:
    workno2filename = json.load(f)

model = SentenceTransformer("/home/user/code/SeqCls/checkpoints/gte-Qwen2-7B-instruct/gte-Qwen2-7B-instruct").half()

for workno, filename in tqdm(workno2filename.items()):
    if workno not in textDict:
        print(workno)
        continue
    text = textDict[workno]
    sentences = sentencesDict[workno]
    response = responseDict[workno]
    splits = []
    for sentence in sentences:
        if len(sentence) < 1024:
            splits.append(sentence)
        else:
            splits += [text[i:i+1024] for i in range(0, len(sentence), 512)]
    splits.append(response)
    embeddings = model.encode(splits, normalize_embeddings=True)
    for f in filename:
        np.save(os.path.join(outputPath, f + ".npy"), embeddings)

