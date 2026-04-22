import os
import pickle

import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from scipy.stats import iqr
from xml.dom import minidom
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

def read_aecg(aecgPath):
    dom = minidom.parse(str(aecgPath))
    sequences = dom.getElementsByTagName('sequence')
    if len(sequences) == 3:
        return False, None, None
    assert (len(sequences) == 2*(8+1) or len(sequences) == 2*(12+1))

    leadNum = len(sequences) // 2 - 1

    code = sequences[0].childNodes[0].getAttribute('code')
    assert (code == 'TIME_ABSOLUTE' or code == 'TIME_RELATIVE')
    # increment = float(sequences[0].childNodes[1].childNodes[1].getAttribute('value'))

    leadData = {}
    origins, units, scales = set(), set(), set()
    for i in range(1, leadNum + 1):
        leadName = sequences[i].childNodes[0].getAttribute('code')
        
        origins.add(sequences[i].childNodes[1].childNodes[0].getAttribute('value'))
        units.add(sequences[i].childNodes[1].childNodes[0].getAttribute('unit'))
        scales.add(sequences[i].childNodes[1].childNodes[1].getAttribute('value'))
        units.add(sequences[i].childNodes[1].childNodes[1].getAttribute('unit'))
        leadData[leadName] = np.array([float(digit) for digit in sequences[i].childNodes[1].childNodes[2].firstChild.data.split()])
    
    if len(origins) == 0 and len(scales) == 0 and len(units) == 0 and len(leadData.keys()) == 0:
        return False, None, None

    assert len(origins) == 1
    assert len(scales) == 1
    assert len(units) == 1

    origin = float(list(origins)[0])
    scale = float(list(scales)[0])
    unit = list(units)[0]

    assert unit == "uV"

    keys = [
        'MDC_ECG_LEAD_I',   
        'MDC_ECG_LEAD_II',  
        'MDC_ECG_LEAD_III', 
        'MDC_ECG_LEAD_aVR', 
        'MDC_ECG_LEAD_aVL', 
        'MDC_ECG_LEAD_aVF', 
        'MDC_ECG_LEAD_V1',
        'MDC_ECG_LEAD_V2',
        'MDC_ECG_LEAD_V3',
        'MDC_ECG_LEAD_V4',
        'MDC_ECG_LEAD_V5',
        'MDC_ECG_LEAD_V6',
        ]

    if leadNum == 8:
        leadData['MDC_ECG_LEAD_III'] = leadData['MDC_ECG_LEAD_II'] - leadData['MDC_ECG_LEAD_I']
        leadData['MDC_ECG_LEAD_aVR'] = - leadData['MDC_ECG_LEAD_II'] / 2.0 - leadData['MDC_ECG_LEAD_I'] / 2.0
        leadData['MDC_ECG_LEAD_aVL'] = leadData['MDC_ECG_LEAD_I'] - leadData['MDC_ECG_LEAD_II'] / 2.0
        leadData['MDC_ECG_LEAD_aVF'] = leadData['MDC_ECG_LEAD_II'] - leadData['MDC_ECG_LEAD_I'] / 2.0

    digits = []
    for i, key in enumerate(keys):
        leadDigits = np.array(leadData[key]) * scale / 1000.0 + origin / 1000.0
        digits.append(leadDigits)
    
    return True, np.array(digits)[:, :5000].T, [key.split('_')[-1] for key in keys]

def resample_data(sigbufs, channel_labels, fs, target_fs, channels=8, channel_stoi=None,skimage_transform=True,interpolation_order=3):
    channel_labels = [c.lower() for c in channel_labels]
    #https://github.com/scipy/scipy/issues/7324 zoom issues
    factor = target_fs/fs
    timesteps_new = int(len(sigbufs)*factor)
    if(channel_stoi is not None):
        data = np.zeros((timesteps_new, channels), dtype=np.float32)
        for i,cl in enumerate(channel_labels):
            if(cl in channel_stoi.keys() and channel_stoi[cl]<channels):
                if(skimage_transform):
                    data[:,channel_stoi[cl]]=transform.resize(sigbufs[:,i],(timesteps_new,),order=interpolation_order).astype(np.float32)
                else:
                    data[:,channel_stoi[cl]]=zoom(sigbufs[:,i],timesteps_new/len(sigbufs),order=interpolation_order).astype(np.float32)
    else:
        if(skimage_transform):
            data=transform.resize(sigbufs,(timesteps_new,channels),order=interpolation_order).astype(np.float32)
        else:
            data=zoom(sigbufs,(timesteps_new/len(sigbufs),1),order=interpolation_order).astype(np.float32)
    return data

def prepare_data_fangchan(data_path, target_fs=100, channels=8, target_folder=None, skimage_transform=True, recreate_data=True):
    target_root_fangchan = Path(".") if target_folder is None else target_folder
    print(target_root_fangchan)
    target_root_fangchan.mkdir(parents=True, exist_ok=True)

    if (recreate_data is True):
        # creating df
        fangchan_meta = []
        fangchanClasses = os.listdir(data_path)
        fangchanClasses.sort()
        for fangchanClass in fangchanClasses:
            worknos = os.listdir(os.path.join(data_path, fangchanClass))
            worknos.sort()
            for workno in worknos:
                ecgs = os.listdir(os.path.join(data_path, fangchanClass, workno))
                ecgs.sort()
                for ecg in ecgs:
                    fangchan_meta.append([workno, fangchanClass, ecg])
        fangchan_meta = np.array(fangchan_meta)
        df_fangchan = pd.DataFrame(fangchan_meta, columns=['workno', 'label', 'filename'])

        fangchan_label_task_all =     {'Normal': 0, 'Prediction':  1,  'Recognition':  2,  'Identification':  3,  'Fangchan':  3}
        fangchan_label_task_pred =    {'Normal': 0, 'Prediction':  1,  'Recognition': -1,  'Identification': -1,  'Fangchan': -1}
        fangchan_label_task_recog =   {'Normal': 0, 'Prediction': -1,  'Recognition':  1,  'Identification': -1,  'Fangchan': -1}
        fangchan_label_task_iden =    {'Normal': 0, 'Prediction': -1,  'Recognition': -1,  'Identification':  1,  'Fangchan': -1}
        fangchan_label_task_combine = {'Normal': 0, 'Prediction':  1,  'Recognition':  2,  'Identification':  3,  'Fangchan': -1}

        df_fangchan["label_all"]     = df_fangchan.label.apply(lambda x: fangchan_label_task_all[x])
        df_fangchan["label_pred"]    = df_fangchan.label.apply(lambda x: fangchan_label_task_pred[x])
        df_fangchan["label_recog"]   = df_fangchan.label.apply(lambda x: fangchan_label_task_recog[x])
        df_fangchan["label_iden"]    = df_fangchan.label.apply(lambda x: fangchan_label_task_iden[x])
        df_fangchan["label_combine"] = df_fangchan.label.apply(lambda x: fangchan_label_task_combine[x])

        df_fangchan["dataset"] = "fangchan"

        lbl_itos_fangchan = {
            "label_all"     : fangchan_label_task_all,
            "label_pred"    : fangchan_label_task_pred,
            "label_recog"   : fangchan_label_task_recog,
            "label_iden"    : fangchan_label_task_iden,
            "label_conbine" : fangchan_label_task_combine
        }

        filenames = []
        drop_idxs = []
        for id, row in tqdm(list(df_fangchan.iterrows())):
            filename = data_path/row["label"]/row["workno"]/row["filename"]
            ret, sigbufs, sig_name = read_aecg(filename)
            # import pdb; pdb.set_trace()
            if ret:
                data = resample_data(sigbufs=sigbufs, 
                                    channel_stoi=channel_stoi, 
                                    channel_labels=sig_name,
                                    fs=500,
                                    target_fs=target_fs,
                                    channels=channels,
                                    skimage_transform=skimage_transform)

                assert target_fs <= 500
                np.save(target_root_fangchan/(filename.stem+".npy"), data)
                filenames.append(Path(filename.stem+".npy"))
            else:
                drop_idxs.append(id)
        df_fangchan = df_fangchan.drop(drop_idxs)
        df_fangchan["data"] = filenames

        # add means and std
        dataset_add_mean_col(df_fangchan, data_folder=target_root_fangchan)
        dataset_add_std_col(df_fangchan, data_folder=target_root_fangchan)
        dataset_add_length_col(df_fangchan, data_folder=target_root_fangchan)
        # dataset_add_median_col(df_fangchan, data_folder=target_root_fangchan)
        # dataset_add_iqr_col(df_fangchan, data_folder=target_root_fangchan)

        # save means and stds
        mean_fangchan, std_fangchan = dataset_get_stats(df_fangchan)

        #save
        save_dataset(df_fangchan, lbl_itos_fangchan, mean_fangchan, std_fangchan, target_root_fangchan)
    else:
        df_fangchan, lbl_itos_fangchan, mean_fangchan, std_fangchan = load_dataset(target_root_fangchan, df_mapped=False)
    return df_fangchan, lbl_itos_fangchan, mean_fangchan, std_fangchan

target_fs = 100
data_root = Path("/home/user/data")
target_root = Path("../ecg_data_processed")

data_folder_fangchan = data_root/"FangChan/ECG/"
target_folder_fangchan = target_root/("fangchan_fs"+str(target_fs))

df_fangchan, lbl_itos_fangchan, mean_fangchan, std_fangchan = prepare_data_fangchan(data_path=data_folder_fangchan, 
                                                                                    target_fs=target_fs, 
                                                                                    channels=12, 
                                                                                    target_folder=target_folder_fangchan)

#reformat everything as memmap for efficiency
reformat_as_memmap(df_fangchan, target_folder_fangchan/("memmap.npy"),data_folder=target_folder_fangchan, delete_npys=False)