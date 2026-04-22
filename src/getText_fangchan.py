import os
import pyarrow.parquet as pq
import numpy as np
import json
from tqdm import tqdm


useful_workno_set = set()
import pickle
with open("/home/user/code/SeqCls/ecg_data_processed/fangchan_fs100/df_memmap.pkl", 'rb') as f:
    df_memmap = pickle.load(f)
for workno in df_memmap['workno']:
    useful_workno_set.add(workno)
    

source = "zhenduan"
parqFolder = "/home/user/code/stat/anzhen2023{}20231201".format(source)
files = os.listdir(parqFolder)
files.sort()
allData = []
for fIdx, fPath in enumerate(tqdm(files, ncols=120)):
    filePath = os.path.join(parqFolder, fPath)
    data = pq.read_table(filePath, columns=['workno', 'uid', 'fname']).to_pandas()
    allData.append(data.values)
allData = np.concatenate(allData, axis=0)
workno2uid, workno2fname = {}, {}
for i in tqdm(range(len(allData)), ncols=120):
    idx = allData[i][0][4:].decode('utf-8')
    if idx not in useful_workno_set:
        continue
    workno2uid[idx] = allData[i][1].decode('utf-8')
    workno2fname[idx] = allData[i][2].decode('utf-8')


with open("/home/user/code/stat/logs/workno2uid.json", 'w') as f:
    json.dump(workno2uid, f)
with open("/home/user/code/stat/logs/workno2fname.json", 'w') as f:
    json.dump(workno2fname, f)
