
import os
import shutil

import numpy as np
import pyarrow.parquet as pq

from tqdm import tqdm
from rich import print
from rich.console import Console
from rich.columns import Columns
from collections import Counter, defaultdict

def processInverseFreq(ids):
    idc = Counter(ids)

    commonNum = 1
    while idc.most_common(commonNum)[-1][1] > 1:
        commonNum *= 2
    
    freq = defaultdict(int)
    notSingleNum = 0
    commonItems = idc.most_common(commonNum)
    for item in commonItems:
        if item[1] > 1:
            freq[item[1]] += 1
            notSingleNum += 1
        else:
            break
    freq[1] = len(idc) - notSingleNum
    return idc, freq

def printCounter(key, counter, ctype='inverse'):
    printRes = []
    if ctype == 'inverse':
        print(key, "(Freq, Count)")
    else:
        print(key, "(Item, Freq)")
    freqs = sorted(list(counter.keys()), key = lambda x : x if x else b'')
    for freq in freqs:
        count = counter[freq]
        printRes.append('|[white]' + str(freq) + '[/white][green]:[/green][blue]' + str(count) + '[/blue]|')
    print(Columns(printRes, align="right"))
    print('Unique %d' % sum(counter.values()))

def getDate(workno):
    year = int(workno[2:6])
    if year > 2000 and year <= 2024:
        return year * 10000 + int(workno[6:10])
    else:
        year = int('20' + str(year)[1:-1])
        if year > 2000 and year <= 2024:
            return year * 10000 + int(workno[5:9])
        else:
            import pdb; pdb.set_trace()
            return None

def getSecondLength(keys, mapping):
    res = 0
    for key in keys:
        values = mapping[key]
        res += len(values)
    return res

def ifExist(query, keys, mapping):
    for key in keys:
        value = mapping[key]
        if value and query in value:
            return True
    return False

def ifExistOther(query, keys, mapping):
    for key in keys:
        value = mapping[key]
        if value and query not in value:
            return True
    return False

def printListInfo(array):
    print(array[0], array[-1], len(array))

def saveList(array, outPath):
    f = open(outPath, 'w')
    for item in array:
        f.write(str(item) + ' ')
    f.close()

def saveAecgList(worknos, folderMapping, inputFolder, outputFolder):

    print("Copying files to", outputFolder)

    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)

    for workno in tqdm(worknos, ncols=120):
        level1Folder = folderMapping[workno]
        if os.path.exists(os.path.join(inputFolder, level1Folder, workno)):
            if not os.path.exists(os.path.join(outputFolder, workno)):
                os.makedirs(os.path.join(outputFolder, workno))
            
            ecgFiles = os.listdir(os.path.join(inputFolder, level1Folder, workno))
            for ecgFile in ecgFiles:
                shutil.copyfile(os.path.join(inputFolder, level1Folder, workno, ecgFile), os.path.join(outputFolder, workno, ecgFile))
        else:
            print(os.path.join(inputFolder, level1Folder, workno))
            # import pdb; pdb.set_trace()

ecgFolder = '/Data-Ecg/EcgUrl/ecgurl20231204'
fangchan = "房颤".encode()

console = Console()

aecgNum, ecgNum = 0, 0

allEcgWorknos = []
nativeFangchanWorknos = []
workno2folder = {}
console.log('Reading ecgs')
level1Folders = os.listdir(ecgFolder)
level1Folders.sort()
for i, level1Folder in enumerate(level1Folders):
    console.log("[%02d]/[%02d]" % (i+1, len(level1Folders)))
    folders = os.listdir(os.path.join(ecgFolder, level1Folder))
    aecgNum += len(folders) - 1

    for item in tqdm(folders, ncols=120):
        if len(item) == 13 and not item.endswith(".txt"):
            allEcgWorknos.append(item)
            workno2folder[item] = level1Folder
        
            filenames = os.listdir(os.path.join(ecgFolder, level1Folder, item))
            for filename in filenames:
                f = open(os.path.join(ecgFolder, level1Folder, item, filename), 'rb')
                data = f.readlines()
                if len(data) == 0:
                    continue
                if fangchan in data[0]:
                    nativeFangchanWorknos.append(item)
                    break
    
    # f = open(os.path.join(ecgFolder, level1Folder, level1Folder + '_data.0.txt'))
    # data = f.readlines()
    # ecgNum += len(data)
    # for item in data:
    #     workno = item.split('\t')[0]
    #     ecgWorknos.append(workno)
    #     workno2folder[workno] = level1Folder

saveList(nativeFangchanWorknos, '../stat/logs/aecgStatementFangchan.txt')

console.log('Loading uid & fname from parquet')
source = "zhenduan"
parqFolder = "../stat/anzhen2023{}20231201".format(source)

files = os.listdir(parqFolder)
files.sort()
allData = []
for fIdx, fPath in enumerate(tqdm(files, ncols=120)):
    filePath = os.path.join(parqFolder, fPath)
    data = pq.read_table(filePath, columns=['workno', 'uid', 'fname']).to_pandas()
    allData.append(data.values)
allData = np.concatenate(allData, axis=0)

console.log('Generating mapping from workno to uid and fname')
workno2uid, workno2fname = {}, {}
for i in tqdm(range(len(allData)), ncols=120):
    idx = allData[i][0][4:].decode('utf-8')
    workno2uid[idx] = allData[i][1]
    workno2fname[idx] = allData[i][2]

console.log('Get mapping from uid to worknos')
# ecgWorknos = list(set(allEcgWorknos))
ecgWorknos = []
ecgWorknoSet = set()
for workno in allEcgWorknos:
    if workno not in ecgWorknoSet:
        ecgWorknos.append(workno)
        ecgWorknoSet.add(workno)

ecgUids = defaultdict(list)
for i in tqdm(range(len(ecgWorknos)), ncols=120):
    uid = workno2uid.get(ecgWorknos[i], None)
    if uid:
        ecgUids[uid].append(ecgWorknos[i])

uidNum = len(ecgUids.keys())

console.log('Get fangchan worknos')

allFangchanWorknos = []
for i in tqdm(range(len(ecgWorknos)), ncols=120):
    fname = workno2fname.get(ecgWorknos[i], None)
    if fname and fangchan in fname:
        allFangchanWorknos.append(ecgWorknos[i])

console.log('Get repeated uids')
repeatUids = []
for i, uid in enumerate(tqdm(ecgUids.keys(), ncols=120)):
    worknos = ecgUids[uid]
    if len(worknos) >= 2:
        repeatUids.append(uid)

console.log('Get repeated & fangchan uids')

fangchanUids = []
normalWorknos = []
for i in tqdm(range(len(repeatUids)), ncols=120):
    worknos = ecgUids[repeatUids[i]]
    flag = ifExist(fangchan, worknos, workno2fname)
    if flag:
        fangchanUids.append(repeatUids[i])
    else:
        normalWorknos += worknos

console.log('Get fanchan & doulv')
includeNormalUids = []
for i in tqdm(range(len(fangchanUids)), ncols=120):
    worknos = ecgUids[fangchanUids[i]]
    flag = ifExistOther(fangchan, worknos, workno2fname)
    if flag:
        includeNormalUids.append(fangchanUids[i])

console.log('Get dates')
predWorknos, recogWorknos, idenWorknos = [], [], []
# predNum, recogNum, idenNum = 0, 0, 0
allStat = []
for i in tqdm(range(len(includeNormalUids)), ncols=120):
    worknos = ecgUids[includeNormalUids[i]]
    fangchanDates, nofangchanDates = [], []
    fangchanWorknos, nofangchanWorknos = [], []
    for workno in worknos:
        date = getDate(workno)
        fname = workno2fname[workno]
        if fname and fangchan in fname:
            fangchanDates.append(date)
            fangchanWorknos.append(workno)
        elif fname and fangchan not in fname:
            nofangchanDates.append(date)
            nofangchanWorknos.append(workno)
    
    dates = fangchanDates + nofangchanDates
    worknos = fangchanWorknos + nofangchanWorknos
    stats = [1] * len(fangchanDates) + [0] * len(nofangchanDates)

    sortedIdxs = sorted(range(len(dates)), key= lambda x : dates[x])
    
    dates = [dates[idx] for idx in sortedIdxs]
    worknos = [worknos[idx] for idx in sortedIdxs]
    stats = [stats[idx] for idx in sortedIdxs]
    
    allStat.append("_".join([str(stat) for stat in stats]))

    idenWorknos += fangchanWorknos
    for j, nofangchanyear in enumerate(nofangchanDates):
        predFlag, recogFlag = 0, 0
        for fangchanyear in fangchanDates:
            if nofangchanyear > fangchanyear and recogFlag == 0:
                recogFlag = 1
            elif nofangchanyear < fangchanyear and predFlag == 0:
                predFlag = 1
            elif recogFlag == 1 or predFlag == 1:
                pass
            else:
                print(worknos)
        
        if predFlag == 1 and recogFlag == 1:
            recogWorknos.append(nofangchanWorknos[j])
        elif predFlag == 1:
            predWorknos.append(nofangchanWorknos[j])
        elif recogFlag == 1:
            recogWorknos.append(nofangchanWorknos[j])
        else:
            print(predFlag, recogFlag, worknos)

    # Another Method        
    # for j in range(len(dates)):
    #     if stats[j] == 1:
    #         idenNum += 1
    #     elif stats[j] == 0:
    #         if 1 in stats[0:j]:
    #             recogNum += 1
    #         if 1 in stats[j+1:len(dates)]:
    #             predNum += 1

allStatCounter = Counter(allStat)
printCounter("AllYear", allStatCounter, 'normal')

print()
console.log("                心电图检查共%d次" % len(ecgWorknos))
# console.log("心电图文件夹共%d份" % aecgNum)
console.log("1--             其中有诊断结果共%d人,%d次" % (uidNum, getSecondLength(ecgUids.keys(), ecgUids)))
console.log("1--1--          其中当次ECG显示房颤共%d次" % len(nativeFangchanWorknos))
console.log("1--2--          其中当次诊断出房颤共%d次 Fangchan" % len(allFangchanWorknos))
console.log("1--3--          其中大于等于两次共%d人,%d次" % (len(repeatUids), getSecondLength(repeatUids, ecgUids)))
console.log("1--3--1--       其中全程正常共%d次 Normal" % len(normalWorknos))
console.log("1--3--2--       其中至少一次有房颤共%d人,%d次" % (len(fangchanUids), getSecondLength(fangchanUids, ecgUids)))
console.log("1--3--2--1--    其中同时有房颤和窦律共%d人,%d次" % (len(includeNormalUids), getSecondLength(includeNormalUids, ecgUids)))
console.log("1--3--2--1--1-- 预测样本共%d次 Prediction" % len(predWorknos))
console.log("1--3--2--1--2-- 识别样本共%d次 Recognition" % len(recogWorknos))
console.log("1--3--2--1--3-- 辨认样本共%d次 Identification" % len(idenWorknos))
print()

# saveAecgList(allFangchanWorknos,   workno2folder, ecgFolder, "//home/user/data/FangChan/ECG/Fangchan"      )
# saveAecgList(normalWorknos[::100], workno2folder, ecgFolder, "//home/user/data/FangChan/ECG/Normal"        )
# saveAecgList(predWorknos,          workno2folder, ecgFolder, "//home/user/data/FangChan/ECG/Prediction"    )
# saveAecgList(recogWorknos,         workno2folder, ecgFolder, "//home/user/data/FangChan/ECG/Recognition"   )
# saveAecgList(idenWorknos,          workno2folder, ecgFolder, "//home/user/data/FangChan/ECG/Identification")

# import pdb; pdb.set_trace()
