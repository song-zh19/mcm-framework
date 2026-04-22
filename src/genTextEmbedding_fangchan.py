import re
import os
import torch

import numpy as np
import pandas as pd
import json

from tqdm import tqdm
from collections import defaultdict
from sentence_transformers import SentenceTransformer


# workno2fnamePath = "/home/user/code/stat/logs/workno2fname.json"
# workno2zhibiaoPath = "/home/user/code/stat/logs/workno2zhibiao_new.json"
# workno2fname = json.load(open(workno2fnamePath, "r"))
# workno2zhibiao = json.load(open(workno2zhibiaoPath, "r"))
# print(f"total number: {len(workno2fname)}")

# textDict = {}
# sentencesDict = {}

# for key in tqdm(workno2fname.keys()):
#     fname = workno2fname[key]
#     zhibiao = workno2zhibiao[key]
#     sentences = []
#     text = "该患者"

#     try:
#         age = int(zhibiao['examage'])
#     except:
#         age = zhibiao['examage']
#     text += str(age) + "岁，"
#     sentences.append("患者" + str(age) + "岁。")

#     gender = zhibiao['examusersex']
#     if gender not in ["男", "女"]:
#         print(key)
#     text += "性别" + gender + "，"
#     sentences.append("患者" + "性别" + gender + "。")

#     iname = zhibiao['iname']
#     iresultvalue = zhibiao['iresultvalue']
#     iunit = zhibiao['iunit']
#     inormallowvalue = zhibiao['inormallowvalue']
#     inormalhighvalue = zhibiao['inormalhighvalue']

#     if len(iunit) == 0:
#         assert len(iname) == 1 and len(iresultvalue) == 1 and len(inormallowvalue) == 0 and len(inormalhighvalue) == 0
#         text += iname[0] + "结果为：" + str(iresultvalue[0]) + "，"
#         sentences.append("患者" + iname[0] + "结果为：" + str(iresultvalue[0]) + "。")
#     else:
#         assert len(iunit) == len(iname) == len(iresultvalue) == len(inormallowvalue) == len(inormalhighvalue)
#         for idx in range(len(iname)):
#             name = iname[idx]
#             resultvalue = iresultvalue[idx]
#             unit = iunit[idx]
#             normallowvalue = inormallowvalue[idx]
#             normalhighvalue = inormalhighvalue[idx]
#             unit = " " + unit if unit != ' ' else ""
#             try:
#                 normalrange = "，正常范围为：" + str(float(normallowvalue)) + "~" + str(float(normalhighvalue)) + unit
#                 if float(resultvalue) < float(normallowvalue):
#                     normalrange += "，结果偏低"
#                 elif float(resultvalue) > float(normalhighvalue):
#                     normalrange += "，结果偏高"
#                 else:
#                     normalrange += "，结果正常"
#             except:
#                 normalrange = ""
#             text += name + "结果为：" + str(resultvalue) + unit + normalrange + "，"
#             sentences.append("患者" + name + "结果为：" + str(resultvalue) + unit + normalrange + "。")

#     text = text[:-1] + "。"
#     text += "患者整体诊断结果为：" + fname + "。"
#     sentences.append("患者整体诊断结果为：" + fname + "。")
#     textDict[key] = text
#     sentencesDict[key] = sentences

# with open("/home/user/code/stat/logs/textDict.json", "w") as f:
#     json.dump(textDict, f)
# with open("/home/user/code/stat/logs/sentencesDict.json", "w") as f:
#     json.dump(sentencesDict, f)


# outputPath = "/home/user/code/SeqCls/ecg_data_processed/fangchan_fs100/acgeEmbedding"
# if not os.path.exists(outputPath):
#     os.makedirs(outputPath)
# # model = SentenceTransformer("/home/user/code/SeqCls/checkpoints/gte-Qwen2-7B-instruct").half()
# model = SentenceTransformer("/home/user/code/SeqCls/checkpoints/acge_text_embedding")

# for key in tqdm(textDict.keys()):
#     text = textDict[key]
#     sentences = sentencesDict[key]
#     splits = []
#     for sentence in sentences:
#         if len(sentence) < 1024:
#             splits.append(sentence)
#         else:
#             splits += [text[i:i+1024] for i in range(0, len(sentence), 512)]
#     embeddings = model.encode(splits, normalize_embeddings=True)
#     np.save(os.path.join(outputPath, key + ".npy"), embeddings)

with open("/home/user/code/stat/logs/textDict.json", "r") as f:
    textDict = json.load(f)
with open("/home/user/code/stat/logs/sentencesDict.json", "r") as f:
    sentencesDict = json.load(f)
outputPath = "/home/user/code/SeqCls/ecg_data_processed/fangchan_fs100/qwen2Embedding"
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
    splits = []
    for sentence in sentences:
        if len(sentence) < 1024:
            splits.append(sentence)
        else:
            splits += [text[i:i+1024] for i in range(0, len(sentence), 512)]
    embeddings = model.encode(splits, normalize_embeddings=True)
    for f in filename:
        np.save(os.path.join(outputPath, f + ".npy"), embeddings)