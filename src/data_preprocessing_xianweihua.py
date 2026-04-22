import os
import json
import scipy
import pickle

import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from scipy.stats import iqr
from skimage import transform
from scipy.ndimage import zoom

channel_stoi = {"i":   0, 
                "ii":  1, 
                "v1":  2, 
                "v2":  3, 
                "v3":  4, 
                "v4":  5, 
                "v5":  6, 
                "v6":  7, 
                "iii": 8, 
                "avr": 9, 
                "avl":10, 
                "avf":11, 
                "vx": 12, 
                "vy": 13, 
                "vz": 14}

def dataset_add_chunk_col(df, col="data"):
    '''add a chunk column to the dataset df'''
    df["chunk"]=df.groupby(col).cumcount()

def dataset_add_length_col(df, col="data", data_folder=None):
    '''add a length column to the dataset df'''
    df[col+"_length"]=df[col].apply(lambda x: len(np.load(x if data_folder is None else data_folder/x, allow_pickle=True)))

def dataset_add_labels_col(df, col="label", data_folder=None):
    '''add a column with unique labels in column col'''
    df[col+"_labels"]=df[col].apply(lambda x: list(np.unique(np.load(x if data_folder is None else data_folder/x, allow_pickle=True))))

def dataset_add_mean_col(df, col="data", axis=(0), data_folder=None):
    '''adds a column with mean'''
    df[col+"_mean"]=df[col].apply(lambda x: np.mean(np.load(x if data_folder is None else data_folder/x, allow_pickle=True),axis=axis))

def dataset_add_median_col(df, col="data", axis=(0), data_folder=None):
    '''adds a column with median'''
    df[col+"_median"]=df[col].apply(lambda x: np.median(np.load(x if data_folder is None else data_folder/x, allow_pickle=True),axis=axis))

def dataset_add_std_col(df, col="data", axis=(0), data_folder=None):
    '''adds a column with mean'''
    df[col+"_std"]=df[col].apply(lambda x: np.std(np.load(x if data_folder is None else data_folder/x, allow_pickle=True),axis=axis))

def dataset_add_iqr_col(df, col="data", axis=(0), data_folder=None):
    '''adds a column with mean'''
    df[col+"_iqr"]=df[col].apply(lambda x: iqr(np.load(x if data_folder is None else data_folder/x, allow_pickle=True),axis=axis))

def dataset_get_stats(df, col="data", simple=True):
    '''creates (weighted) means and stds from mean, std and length cols of the df'''
    if(simple):
        return df[col+"_mean"].mean(), df[col+"_std"].mean()
    else:
        #https://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html
        #or https://gist.github.com/thomasbrandon/ad5b1218fc573c10ea4e1f0c63658469
        def combine_two_means_vars(x1,x2):
            (mean1,var1,n1) = x1
            (mean2,var2,n2) = x2
            mean = mean1*n1/(n1+n2)+ mean2*n2/(n1+n2)
            var = var1*n1/(n1+n2)+ var2*n2/(n1+n2)+n1*n2/(n1+n2)/(n1+n2)*np.power(mean1-mean2,2)
            return (mean, var, (n1+n2))

        def combine_all_means_vars(means,vars,lengths):
            inputs = list(zip(means,vars,lengths))
            result = inputs[0]

            for inputs2 in inputs[1:]:
                result= combine_two_means_vars(result,inputs2)
            return result

        means = list(df[col+"_mean"])
        vars = np.power(list(df[col+"_std"]),2)
        lengths = list(df[col+"_length"])
        mean,var,length = combine_all_means_vars(means,vars,lengths)
        return mean, np.sqrt(var)

def save_dataset(df,lbl_itos,mean,std,target_root,filename_postfix="",protocol=4):
    target_root = Path(target_root)
    df.to_pickle(target_root/("df"+filename_postfix+".pkl"), protocol=protocol)

    if(isinstance(lbl_itos,dict)):#dict as pickle
        outfile = open(target_root/("lbl_itos"+filename_postfix+".pkl"), "wb")
        pickle.dump(lbl_itos, outfile, protocol=protocol)
        outfile.close()
    else:#array
        np.save(target_root/("lbl_itos"+filename_postfix+".npy"),lbl_itos)

    np.save(target_root/("mean"+filename_postfix+".npy"),mean)
    np.save(target_root/("std"+filename_postfix+".npy"),std)

def load_dataset(target_root,filename_postfix="",df_mapped=True):
    target_root = Path(target_root)
    # if(df_mapped):
    #     df = pd.read_pickle(target_root/("df_memmap"+filename_postfix+".pkl"))
    # else:
    #     df = pd.read_pickle(target_root/("df"+filename_postfix+".pkl")
    
    ### due to pickle 5 protocol error

    if(df_mapped):
        df = pickle.load(open(target_root/("df_memmap"+filename_postfix+".pkl"), "rb"))
    else:
        df = pickle.load(open(target_root/("df"+filename_postfix+".pkl"), "rb"))


    if((target_root/("lbl_itos"+filename_postfix+".pkl")).exists()):#dict as pickle
        infile = open(target_root/("lbl_itos"+filename_postfix+".pkl"), "rb")
        lbl_itos=pickle.load(infile)
        infile.close()
    else:#array
        lbl_itos = np.load(target_root/("lbl_itos"+filename_postfix+".npy"))


    mean = np.load(target_root/("mean"+filename_postfix+".npy"))
    std = np.load(target_root/("std"+filename_postfix+".npy"))
    return df, lbl_itos, mean, std

def npys_to_memmap_batched(npys, target_filename, max_len=0, delete_npys=True, batch_length=900000):
    memmap = None
    start = np.array([0])#start_idx in current memmap file (always already the next start- delete last token in the end)
    length = []#length of segment
    filenames= []#memmap files
    file_idx=[]#corresponding memmap file for sample
    shape=[]#shapes of all memmap files

    data = []
    data_lengths=[]
    dtype = None

    for idx,npy in tqdm(list(enumerate(npys))):

        data.append(np.load(npy, allow_pickle=True))
        data_lengths.append(len(data[-1]))

        if(idx==len(npys)-1 or np.sum(data_lengths)>batch_length):#flush
            data = np.concatenate(data)
            if(memmap is None or (max_len>0 and start[-1]>max_len)):#new memmap file has to be created
                if(max_len>0):
                    filenames.append(target_filename.parent/(target_filename.stem+"_"+str(len(filenames))+".npy"))
                else:
                    filenames.append(target_filename)

                shape.append([np.sum(data_lengths)]+[l for l in data.shape[1:]])#insert present shape

                if(memmap is not None):#an existing memmap exceeded max_len
                    del memmap
                #create new memmap
                start[-1] = 0
                start = np.concatenate([start,np.cumsum(data_lengths)])
                length = np.concatenate([length,data_lengths])

                memmap = np.memmap(filenames[-1], dtype=data.dtype, mode='w+', shape=data.shape)
            else:
                #append to existing memmap
                start = np.concatenate([start,start[-1]+np.cumsum(data_lengths)])
                length = np.concatenate([length,data_lengths])
                shape[-1] = [start[-1]]+[l for l in data.shape[1:]]
                memmap = np.memmap(filenames[-1], dtype=data.dtype, mode='r+', shape=tuple(shape[-1]))

            #store mapping memmap_id to memmap_file_id
            file_idx=np.concatenate([file_idx,[(len(filenames)-1)]*len(data_lengths)])
            #insert the actual data
            memmap[start[-len(data_lengths)-1]:start[-len(data_lengths)-1]+len(data)]=data[:]
            memmap.flush()
            dtype = data.dtype
            data = []#reset data storage
            data_lengths = []

    start= start[:-1]#remove the last element
    #cleanup
    for npy in npys:
        if(delete_npys is True):
            npy.unlink()
    del memmap

    #convert everything to relative paths
    filenames= [f.name for f in filenames]
    #save metadata
    np.savez(target_filename.parent/(target_filename.stem+"_meta.npz"),start=start,length=length,shape=shape,file_idx=file_idx,dtype=dtype,filenames=filenames)


def npys_to_memmap(npys, target_filename, max_len=0, delete_npys=True):
    memmap = None
    start = []#start_idx in current memmap file
    length = []#length of segment
    filenames= []#memmap files
    file_idx=[]#corresponding memmap file for sample
    shape=[]

    for idx,npy in tqdm(list(enumerate(npys))):
        data = np.load(npy, allow_pickle=True)
        if(memmap is None or (max_len>0 and start[-1]+length[-1]>max_len)):
            if(max_len>0):
                filenames.append(target_filename.parent/(target_filename.stem+"_"+str(len(filenames)+".npy")))
            else:
                filenames.append(target_filename)

            if(memmap is not None):#an existing memmap exceeded max_len
                shape.append([start[-1]+length[-1]]+[l for l in data.shape[1:]])
                del memmap
            #create new memmap
            start.append(0)
            length.append(data.shape[0])
            memmap = np.memmap(filenames[-1], dtype=data.dtype, mode='w+', shape=data.shape)
        else:
            #append to existing memmap
            start.append(start[-1]+length[-1])
            length.append(data.shape[0])
            memmap = np.memmap(filenames[-1], dtype=data.dtype, mode='r+', shape=tuple([start[-1]+length[-1]]+[l for l in data.shape[1:]]))

        #store mapping memmap_id to memmap_file_id
        file_idx.append(len(filenames)-1)
        #insert the actual data
        memmap[start[-1]:start[-1]+length[-1]]=data[:]
        memmap.flush()
        if(delete_npys is True):
            npy.unlink()
    del memmap

    #append final shape if necessary
    if(len(shape)<len(filenames)):
        shape.append([start[-1]+length[-1]]+[l for l in data.shape[1:]])
    #convert everything to relative paths
    filenames= [f.name for f in filenames]
    #save metadata
    np.savez(target_filename.parent/(target_filename.stem+"_meta.npz"),start=start,length=length,shape=shape,file_idx=file_idx,dtype=data.dtype,filenames=filenames)


def reformat_as_memmap(df, target_filename, data_folder=None, annotation=False, max_len=0, delete_npys=True,col_data="data",col_label="label", batch_length=0):
    npys_data = []
    npys_label = []

    for id,row in df.iterrows():
        npys_data.append(data_folder/row[col_data] if data_folder is not None else row[col_data])
        if(annotation):
            npys_label.append(data_folder/row[col_label] if data_folder is not None else row[col_label])
    if(batch_length==0):
        npys_to_memmap(npys_data, target_filename, max_len=max_len, delete_npys=delete_npys)
    else:
        npys_to_memmap_batched(npys_data, target_filename, max_len=max_len, delete_npys=delete_npys,batch_length=batch_length)
    if(annotation):
        if(batch_length==0):
            npys_to_memmap(npys_label, target_filename.parent/(target_filename.stem+"_label.npy"), max_len=max_len, delete_npys=delete_npys)
        else:
            npys_to_memmap_batched(npys_label, target_filename.parent/(target_filename.stem+"_label.npy"), max_len=max_len, delete_npys=delete_npys, batch_length=batch_length)

    #replace data(filename) by integer
    df_mapped = df.copy()
    df_mapped["data_original"]=df_mapped.data
    df_mapped["data"]=np.arange(len(df_mapped))

    df_mapped.to_pickle(target_filename.parent/("df_"+target_filename.stem+".pkl"))
    return df_mapped

def interpSort(points):
    points = np.array(points)
    f = scipy.interpolate.interp1d(points[:, 0], points[:, 1])
    newx = np.linspace(points[:, 0].min(), points[:, 0].max(), 250)
    newy = f(newx)

    return np.concatenate([newx[:, np.newaxis], newy[:, np.newaxis]], axis=1)

def read_json(jsonPath, scaleDict):
    # read annotations
    with open(jsonPath, 'rb') as f:
        annotations = json.load(f) 
    allChannelPoints = annotations['datasetColl']
    
    # resample signals
    resample_signals = []
    for channelIdx, channelPoints in enumerate(allChannelPoints):
        points = channelPoints['data']
        points = [point['value'] for point in points]
        points = interpSort(points)
        resample_signals.append(points)
    
    resample_signals = np.array(resample_signals)
    resample_signals[6:, :, 0] = resample_signals[6:, :, 0] - resample_signals[6:, :1, 0]
    resample_signals[6:, :, 1] = resample_signals[6:, :, 1] - resample_signals[:6, :1, 1]
    resample_signals[:6] = resample_signals[:6] - resample_signals[:6, :1]

    scaleKey = str(jsonPath).split('/')[-2] + '/' + str(jsonPath).split('/')[-1]
    scales = scaleDict.get(scaleKey, [10, 10])
    resample_signals[:, :, 0] = resample_signals[:, :, 0] * 0.2
    resample_signals[:6, :, 1] = resample_signals[:6, :, 1] * 10 / (2 * scales[0])
    resample_signals[6:, :, 1] = resample_signals[6:, :, 1] * 10 / (2 * scales[1])

    # maxTimes = resample_signals[:, -1, 0]
    # if max(maxTimes) > 5.25 or min(maxTimes) < 4.75:
    #     import pdb; pdb.set_trace()
    
    return resample_signals[:, :, 1].T

def resample_data(sigbufs, channel_labels, channels=8, channel_stoi=None):
    channel_labels = [c.lower() for c in channel_labels]
    if(channel_stoi is not None):
        data = np.zeros_like(sigbufs, dtype=np.float32)
        for i,cl in enumerate(channel_labels):
            if(cl in channel_stoi.keys() and channel_stoi[cl]<channels):
                data[:,channel_stoi[cl]] = sigbufs[:,i].astype(np.float32)
    else:
        data = sigbufs.astype(np.float32)
    return data

def prepare_data_xianweihua(data_path, submap_path, scale_path, channels=8, target_folder=None, recreate_data=True):
    target_root_xianweihua = Path(".") if target_folder is None else target_folder
    print(target_root_xianweihua)
    target_root_xianweihua.mkdir(parents=True, exist_ok=True)

    with open(submap_path, 'r') as f:
        name2labels = json.load(f)

    if (recreate_data is True):
        # creating df
        xianweihua_meta = []
        xianweihuaClasses = os.listdir(data_path)
        xianweihuaClasses.sort()
        for xianweihuaClass in xianweihuaClasses:
            files = os.listdir(os.path.join(data_path, xianweihuaClass))
            files.sort()
            if "desktop.ini" in files:
                files.remove("desktop.ini")
            for file in files:
                if xianweihuaClass == 'NMR+':
                    labels = name2labels.get(file, '-1')
                else:
                    labels = '0'
                xianweihua_meta.append([xianweihuaClass, labels, file])
        xianweihua_meta = np.array(xianweihua_meta)
        df_xianweihua = pd.DataFrame(xianweihua_meta, columns=['label', 'label_part', 'filename'])

        xianweihua_label_super = {'NMR-': 0, 'NMR+': 1}
        xianweihua_label_sub = {'0':0, '1':1, '2':2, '3':3, '4':4, '5':5, '6':6}
        df_xianweihua["label_super"] = df_xianweihua.label.apply(lambda x: xianweihua_label_super[x])
        df_xianweihua["label_sub"] = df_xianweihua.label_part.apply(lambda x: [xianweihua_label_sub.get(y, -1) for y in x.split(',')])
        df_xianweihua["dataset"] = "xianweihua"

        lbl_itos_xianweihua = {
            "label_super": xianweihua_label_super,
            'label_sub': xianweihua_label_sub
            }

        with open(scale_path, 'r') as f:
            scaleDict = json.load(f)
        
        sig_name = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']

        filenames = []
        for id, row in tqdm(list(df_xianweihua.iterrows())):
            filename = data_path/row["label"]/row["filename"]
            sigbufs = read_json(filename, scaleDict)
            data = resample_data(sigbufs=sigbufs, 
                                channel_stoi=channel_stoi, 
                                channel_labels=sig_name,
                                channels=channels)

            np.save(target_root_xianweihua/(filename.stem+".npy"), data)
            filenames.append(Path(filename.stem+".npy"))
        df_xianweihua["data"] = filenames

        # add means and std
        dataset_add_mean_col(df_xianweihua, data_folder=target_root_xianweihua)
        dataset_add_std_col(df_xianweihua, data_folder=target_root_xianweihua)
        dataset_add_length_col(df_xianweihua, data_folder=target_root_xianweihua)
        # dataset_add_median_col(df_xianweihua, data_folder=target_root_xianweihua)
        # dataset_add_iqr_col(df_xianweihua, data_folder=target_root_xianweihua)

        # save means and stds
        mean_xianweihua, std_xianweihua = dataset_get_stats(df_xianweihua)

        #save
        save_dataset(df_xianweihua, lbl_itos_xianweihua, mean_xianweihua, std_xianweihua, target_root_xianweihua)
    else:
        df_xianweihua, lbl_itos_xianweihua, mean_xianweihua, std_xianweihua = load_dataset(target_root_xianweihua, df_mapped=False)
    return df_xianweihua, lbl_itos_xianweihua, mean_xianweihua, std_xianweihua

data_root=Path("../ecg_data/")
target_root=Path("../ecg_data_processed")

data_folder_xianweihua = data_root/"xianweihua/Annos3/"
target_folder_xianweihua = target_root/"xianweihua"
scale_path = data_root/"xianweihua/scale.json"
submap_path = data_root/"xianweihua/name2labels.json"

df_xianweihua, lbl_itos_xianweihua, mean_xianweihua, std_xianweihua = prepare_data_xianweihua(data_path=data_folder_xianweihua, 
                                                                                              scale_path=scale_path,
                                                                                              submap_path=submap_path,
                                                                                    channels=12, 
                                                                                    target_folder=target_folder_xianweihua)

#reformat everything as memmap for efficiency
reformat_as_memmap(df_xianweihua, target_folder_xianweihua/("memmap.npy"),data_folder=target_folder_xianweihua, delete_npys=False)