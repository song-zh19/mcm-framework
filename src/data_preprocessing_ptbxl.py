from pathlib import Path

from clinical_ts.ecg_utils import prepare_data_ptb_xl, channel_stoi_default
from clinical_ts.timeseries_utils import reformat_as_memmap

target_fs=100
data_root=Path("../ecg_data/")
target_root=Path("../ecg_data_processed")

# Download the PTB-XL dataset (https://www.nature.com/articles/s41597-020-0495-6) 
# https://physionet.org/content/ptb-xl/1.0.1/ 
# and place it in data_folder/ptb_xl

data_folder_ptb_xl = data_root/"ptb_xl/"
target_folder_ptb_xl = target_root/("ptb_xl_fs"+str(target_fs))

df_ptb_xl, lbl_itos_ptb_xl,  mean_ptb_xl, std_ptb_xl = prepare_data_ptb_xl(data_folder_ptb_xl, min_cnt=0, target_fs=target_fs, channels=12, channel_stoi=channel_stoi_default, target_folder=target_folder_ptb_xl)

#reformat everything as memmap for efficiency
reformat_as_memmap(df_ptb_xl, target_folder_ptb_xl/("memmap.npy"),data_folder=target_folder_ptb_xl,delete_npys=True)