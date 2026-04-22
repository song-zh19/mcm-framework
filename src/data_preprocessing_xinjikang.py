import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
from scipy.stats import iqr
from skimage import transform
from scipy.ndimage import zoom

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

def npys_to_memmap(npys, target_filename, max_len=0, delete_npys=True):
    memmap = None
    start = []#start_idx in current memmap file
    length = []#length of segment
    filenames= []#memmap files
    file_idx=[]#corresponding memmap file for sample
    shape=[]

    cur_npy_name = None
    npy_data = None

    for idx,npy in tqdm(list(enumerate(npys))):
        if '@' in str(npy):
            npy_name, npy_idx = npy.name.split('@')
            if npy_name != cur_npy_name:
                npy_data = np.load(npy.parent/npy_name, allow_pickle=True)
                cur_npy_name = npy_name
            data = npy_data[int(npy_idx)]
        else:
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


def reformat_as_memmap(df, target_filename, data_folder=None, annotation=False, max_len=0, delete_npys=True,col_data="data",col_label="label"):
    #replace data(filename) by integer
    df_mapped = df.copy()
    df_mapped["data_original"]=df_mapped.data
    df_mapped["data"]=np.arange(len(df_mapped))

    df_mapped.to_pickle(target_filename.parent/("df_"+target_filename.stem+".pkl"))
    return df_mapped

def resample_data(sigbufs, channel_labels, fs, target_fs, channels=8, channel_stoi=None,skimage_transform=True,interpolation_order=1):
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

def prepare_data_xinjikang(data_path, target_fs=100, channels=8, target_folder=None, skimage_transform=True, recreate_data=True):
    target_root_xinjikang = Path(".") if target_folder is None else target_folder
    print(target_root_xinjikang)
    target_root_xinjikang.mkdir(parents=True, exist_ok=True)

    if (recreate_data is True):
        # creating df
        xinjikang_meta = []
        for xinjikangPosition in ['Anzhen', 'Quzhou']:
            if xinjikangPosition == 'Anzhen':
                xinjikangClasses = ['2022/NORMAL', '2022/PAF', '2024/NORMAL', '2024/PAF/原始数据']
            else:
                xinjikangClasses = ['NORMAL', 'PAF']
            for xinjikangClass in xinjikangClasses:
                sig_names = os.listdir(os.path.join(data_path, xinjikangPosition, xinjikangClass))
                sig_names.sort()
                print('Processing', data_path/xinjikangPosition/xinjikangClass)
                for sig_name in tqdm(sig_names):
                    if sig_name.endswith('.ehr'):
                        continue
                    sig_file = data_path/xinjikangPosition/xinjikangClass/sig_name
                    # print(sig_file)
                    try:
                        sigbufs = np.loadtxt(sig_file, dtype=np.float32)
                    except:
                        print(sig_file)
                        continue

                    if sigbufs.min() == 0 and sigbufs.max() == 4095:
                        sigbufs = sigbufs - 2048
                    elif sigbufs.min() == -2048 and sigbufs.max() == 2047:
                        sigbufs = sigbufs
                    else:
                        print(sigbufs.min(), sigbufs.max())
                        if xinjikangPosition == 'Quzhou':
                            sigbufs = sigbufs
                        else:
                            sigbufs = sigbufs - 2048


                    fs = 256

                    for idx in range(fs * 30, len(sigbufs), fs * 10):
                        sample = sigbufs[(idx-fs*30):idx]
                        sample_std = np.std(sample)
                        if sample_std != 0:
                            sample_outname = sig_file.stem + '_' + str(idx-fs*30)
                            label = "potentialPAF" if xinjikangPosition == 'Anzhen' else 'NORMAL'
                            xinjikang_meta.append([sample_outname, label, str(sig_file) + "@" + str(idx-fs*30) + '_' + str(idx), np.mean(sample), sample_std, len(sample)])
                    
                    # break
                # break
            # break
        
        xinjikang_meta = np.array(xinjikang_meta)
        df_xinjikang = pd.DataFrame(xinjikang_meta, 
                                     columns=['workno', 'label', 'data', 'data_mean', 'data_std', 'data_length'])

        df_xinjikang[['data_mean', 'data_std', 'data_length']] = df_xinjikang[['data_mean', 'data_std', 'data_length']].apply(pd.to_numeric)

        xinjikang_label_task_all = {'NORMAL': 0, 'potentialPAF':  1}
        df_xinjikang["label_all"] = df_xinjikang.label.apply(lambda x: xinjikang_label_task_all[x])
        df_xinjikang["dataset"] = "xinjikang"

        lbl_itos_xinjikang = {"label_all" : xinjikang_label_task_all}

        # save means and stds
        mean_xinjikang, std_xinjikang = dataset_get_stats(df_xinjikang)

        #save
        save_dataset(df_xinjikang, lbl_itos_xinjikang, mean_xinjikang, std_xinjikang, target_root_xinjikang)
    else:
        df_xinjikang, lbl_itos_xinjikang, mean_xinjikang, std_xinjikang = load_dataset(target_root_xinjikang, df_mapped=False)
    return df_xinjikang, lbl_itos_xinjikang, mean_xinjikang, std_xinjikang

target_fs = 64
data_root=Path("../ecg_data/")
npy_root=Path("../ecg_data_processed")
memmap_root=Path("../ecg_data_processed")

data_folder_xinjikang = data_root/"azfangchan"
target_folder_xinjikang = npy_root/("xinjikang_fs"+str(target_fs))
memmap_folder_xinjikang = memmap_root/("xinjikang_fs"+str(target_fs))

df_xinjikang, lbl_itos_xinjikang, mean_xinjikang, std_xinjikang = prepare_data_xinjikang(data_path=data_folder_xinjikang, 
                                                                                    target_fs=target_fs, 
                                                                                    channels=1, 
                                                                                    target_folder=target_folder_xinjikang)

#reformat everything as memmap for efficiency
reformat_as_memmap(df_xinjikang, target_folder_xinjikang/("memmap.npy"), data_folder=target_folder_xinjikang, delete_npys=False)