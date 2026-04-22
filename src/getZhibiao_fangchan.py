import os
import pyarrow.parquet as pq
import numpy as np
import json
from tqdm import tqdm
import pandas as pd


zhibiao = pd.read_parquet("/home/user/code/stat/anzhen2023zhibiao20231201/refine_zhibiao.parq")

workno2zhibiao = {}
keys = ['examusersex', 'examage', 'iname', 'iresultvalue', 'iunit', 'inormalhighvalue', 'inormallowvalue']

for i in tqdm(range(len(zhibiao)), ncols=120):
    workno = zhibiao['workno'][i][4:].decode('utf-8')
    workno2zhibiao[workno] = {}
    for key in keys:
        try:
            value = zhibiao[key][i].decode('utf-8')
        except:
            value = zhibiao[key][i]
        if key not in ['examusersex', 'examage']:
            if value is None:
                workno2zhibiao[workno][key] = []
            else:
                workno2zhibiao[workno][key] = value.split(',')
        else:
            workno2zhibiao[workno][key] = value

with open("/home/user/code/stat/logs/workno2zhibiao.json", 'w') as f:
    json.dump(workno2zhibiao, f)
