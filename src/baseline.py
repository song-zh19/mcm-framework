###############
#generic
import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data._utils.collate import default_collate
from torchvision import transforms
import torch.nn.functional as F

import os
import argparse
from loguru import logger
from sklearn.model_selection import train_test_split
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import copy

#################
#specific
from clinical_ts.timeseries_utils import *
from clinical_ts.ecg_utils import *

from functools import partial
from pathlib import Path
import pandas as pd
import numpy as np

from clinical_ts.basic_conv1d import weight_init
from clinical_ts.eval_utils_cafa import eval_scores, eval_scores_bootstrap
from clinical_ts.cpc import *

from clinical_ts.fcn_wang import fcn_wang
from clinical_ts.bi_lstm import lstm, lstm_bidir
from clinical_ts.resnet1d_wang import resnet1d_wang
from clinical_ts.inceptiontime import inceptiontime
from clinical_ts.xresnet1d101 import xresnet1d101
from clinical_ts.mobilenet_v3 import mobilenetv3_large, mobilenetv3_small
from clinical_ts.acnet import acnet
from clinical_ts.ati_cnn import ATI_CNN
from clinical_ts.MVMnet import MVMnet


class LightningBaseline(pl.LightningModule):

    def __init__(self, hparams):
        super(LightningBaseline, self).__init__()
        
        # self.hparams = hparams
        self.save_hyperparameters(hparams)
        self.lr = self.hparams.lr
        self.best_metric = None

        assert self.hparams.finetune    # finetune all baseline models

        if (self.hparams.finetune_dataset == "thew" or self.hparams.finetune_dataset in ['fangchan_all', 'fangchan_combine']):
            self.criterion = F.cross_entropy
        else:
            self.criterion = F.binary_cross_entropy_with_logits

        if (self.hparams.finetune_dataset == "thew"):
            self.num_classes = 5
        elif (self.hparams.finetune_dataset == "ptbxl_super"):
            self.num_classes = 5
        elif (self.hparams.finetune_dataset == "ptbxl_all"):
            self.num_classes = 71
        elif (self.hparams.finetune_dataset == "xianweihua_super"):
            self.num_classes = 1
        elif (self.hparams.finetune_dataset == "xianweihua_sub"):
            self.num_classes = 7
        elif (self.hparams.finetune_dataset in ["fangchan_all", "fangchan_combine"]):
            self.num_classes = 4
        elif (self.hparams.finetune_dataset in ["fangchan_pred", "fangchan_recog", "fangchan_iden"]):
            self.num_classes = 1
        elif (self.hparams.finetune_dataset == "xinjikang_all"):
            self.num_classes = 1
        elif (self.hparams.finetune_dataset == "dandaolian_all"):
            self.num_classes = 1

        if self.hparams.model == "fcn_wang":
            self.model = fcn_wang(input_channels=self.hparams.input_channels, num_classes=self.num_classes)
        elif self.hparams.model == "lstm":
            self.model = lstm(num_classes=self.num_classes, input_channels=self.hparams.input_channels)
        elif self.hparams.model == "lstm_bidir":
            self.model = lstm_bidir(num_classes=self.num_classes, input_channels=self.hparams.input_channels)
        elif self.hparams.model == "resnet1d_wang":
            self.model = resnet1d_wang(num_classes=self.num_classes, input_channels=self.hparams.input_channels)
        elif self.hparams.model == "inceptiontime":
            self.model = inceptiontime(num_classes=self.num_classes, in_channel=self.hparams.input_channels, multi_modal_dim=self.hparams.modal_dim if self.hparams.multi_modal else None, n_hidden=self.hparams.n_hidden, lin_ftrs_head=eval(self.hparams.lin_ftrs_head), ps_head=self.hparams.dropout_head)
        elif self.hparams.model == "xresnet1d101":
            self.model = xresnet1d101(num_classes=self.num_classes, input_channels=self.hparams.input_channels)
        elif self.hparams.model == "mobilenetv3_large":
            self.model = mobilenetv3_large(num_classes=self.num_classes, in_channel=self.hparams.input_channels)
        elif self.hparams.model == "mobilenetv3_small":
            self.model = mobilenetv3_small(num_classes=self.num_classes, in_channel=self.hparams.input_channels)
        elif self.hparams.model == "acnet":
            self.model = acnet(input_channels=self.hparams.input_channels, num_classes=self.num_classes)
        elif self.hparams.model == "ati_cnn":
            self.model = ATI_CNN(input_channels=self.hparams.input_channels, num_classes=self.num_classes)
        elif self.hparams.model == "MVMnet":
            self.model = MVMnet(num_classes=self.num_classes)
        else:
            raise NotImplementedError
        
    def configure_optimizers(self):
        if(self.hparams.optimizer == "sgd"):
            opt = torch.optim.SGD
        elif(self.hparams.optimizer == "adam"):
            opt = torch.optim.AdamW
        else:
            raise NotImplementedError("Unknown Optimizer.")

        return opt(self.parameters(), lr=self.lr, weight_decay=self.hparams.weight_decay)
    
    def load_weights_from_checkpoint(self, checkpoint):
        """ Function that loads the weights from a given checkpoint file. 
        based on https://github.com/PyTorchLightning/pytorch-lightning/issues/525
        """
        checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage,)
        pretrained_dict = checkpoint["state_dict"]
        model_dict = self.state_dict()
            
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def setup(self, stage):
        # configure dataset params
        chunkify_train = False
        chunk_length_train = self.hparams.input_size if chunkify_train else 0
        stride_train = self.hparams.input_size//2
        
        chunkify_valtest = False
        chunk_length_valtest = self.hparams.input_size if chunkify_valtest else 0
        stride_valtest = self.hparams.input_size//2

        train_datasets = []
        val_datasets = []
        test_datasets = []
        
        for i, (target_folder, data_folder) in enumerate(zip(self.hparams.meta, self.hparams.data)):
            target_folder = Path(target_folder)     
            data_folder = Path(data_folder)      
            
            df_mapped, lbl_itos,  mean, std = load_dataset(target_folder)
            # always use PTB-XL stats
            mean = np.array([-0.00184586, -0.00130277,  0.00017031, -0.00091313, -0.00148835,  -0.00174687, -0.00077071, -0.00207407,  0.00054329,  0.00155546,  -0.00114379, -0.00035649])
            std = np.array([0.16401004, 0.1647168 , 0.23374124, 0.33767231, 0.33362807,  0.30583013, 0.2731171 , 0.27554379, 0.17128962, 0.14030828,   0.14606956, 0.14656108])

            # specific for PTB-XL
            if(self.hparams.finetune and self.hparams.finetune_dataset.startswith("ptbxl")):
                if(self.hparams.finetune_dataset=="ptbxl_super"):
                    ptb_xl_label = "label_diag_superclass"
                elif(self.hparams.finetune_dataset=="ptbxl_all"):
                    ptb_xl_label = "label_all"
                    
                lbl_itos= np.array(lbl_itos[ptb_xl_label])
                
                def multihot_encode(x, num_classes):
                    res = np.zeros(num_classes,dtype=np.float32)
                    for y in x:
                        res[y]=1
                    return res
                    
                df_mapped["label"]= df_mapped[ptb_xl_label+"_filtered_numeric"].apply(lambda x: multihot_encode(x,len(lbl_itos)))
                    
            if self.hparams.finetune_dataset.startswith("fangchan"):
                task = self.hparams.finetune_dataset.split('_')[-1]
                df_mapped = df_mapped[df_mapped['label_' + task] >= 0]

                max_count_other_classes = df_mapped[df_mapped['label'] != 'Normal']['label'].value_counts().max()

                # 决定保留特定类数量的倍数
                n = 1

                normal_df = df_mapped[df_mapped['label'] == 'Normal']
                required_count = max_count_other_classes * n
                if len(normal_df) > required_count:
                    normal_df = normal_df.iloc[:required_count]
                df_mapped = pd.concat([df_mapped[df_mapped['label'] != 'Normal'], normal_df])

                df_mapped['label'] = df_mapped['label_' + task]

                lbl_itos = df_mapped['label_' + task].unique()

                if (self.hparams.finetune_dataset in ["fangchan_pred", "fangchan_recog", "fangchan_iden"]):
                    df_mapped['label'] = df_mapped['label'].apply(lambda x: np.array([float(x)]))
                
                df_train, df_val = [], []
                for cls in lbl_itos:
                    class_df = df_mapped[df_mapped['label_' + task] == cls]
                    class_train, class_test = train_test_split(class_df, test_size=0.2, random_state=666)
                    df_train.append(class_train)
                    df_val.append(class_test)
                df_train = pd.concat(df_train)
                df_val = pd.concat(df_val)
                if(self.hparams.finetune):
                    df_test = df_val
            
            elif self.hparams.finetune_dataset.startswith("xianweihua"):
                task = self.hparams.finetune_dataset.split('_')[-1]

                def multihot_encode(x, num_classes):
                    res = np.zeros(num_classes,dtype=np.float32)
                    for y in x:
                        res[y]=1
                    return res

                if task == 'super':
                    lbl_itos = df_mapped['label_super'].unique()
                    df_mapped['label'] = df_mapped['label_super'].apply(lambda x: np.array([float(x)]))
                    
                    df_train, df_val = [], []
                    for cls in lbl_itos:
                        class_df = df_mapped[df_mapped['label_super'] == cls]
                        class_train, class_test = train_test_split(class_df, test_size=0.2, random_state=666)
                        df_train.append(class_train)
                        df_val.append(class_test)
                    df_train = pd.concat(df_train)
                    df_val = pd.concat(df_val)
                elif task == 'sub':
                    lbl_itos = np.array(list(lbl_itos['label_sub'].keys()))
                    df_mapped = df_mapped[df_mapped['label_part'] != '-1']
                    df_mapped["label"] = df_mapped["label_sub"].apply(lambda x: multihot_encode(x, len(lbl_itos)))

                    normal_df = df_mapped[df_mapped['label_part'] == '0']
                    normal_train, normal_val = train_test_split(normal_df, test_size=0.2, random_state=666)

                    abnormal_df = df_mapped[df_mapped['label_part'] != '0']
                    abnormal_train, abnormal_val = train_test_split(abnormal_df, test_size=0.2, random_state=666)

                    df_train = pd.concat([normal_train, abnormal_train])
                    df_val = pd.concat([normal_val, abnormal_val])
                else:
                    raise NotImplementedError()

                if(self.hparams.finetune):
                    df_test = df_val
            
            elif self.hparams.finetune_dataset.startswith("xinjikang"):
                logger.info('Processing ' + self.hparams.finetune_dataset)
                task = self.hparams.finetune_dataset.split('_')[-1]
                # df_mapped = df_mapped[df_mapped['label_' + task] >= 0]

                # df_mapped = pd.concat([df_mapped.iloc[0:500000], df_mapped.iloc[-500000:]])

                max_count_other_classes = df_mapped[df_mapped['label'] != 'NORMAL']['label'].value_counts().max()

                # 决定保留特定类数量的倍数
                n = 1

                if n > 0:
                    normal_df = df_mapped[df_mapped['label'] == 'NORMAL']
                    required_count = max_count_other_classes * n
                    logger.info('Remain count of NORMAL samples: ' + str(required_count))
                    if len(normal_df) > required_count:
                        normal_df = normal_df.iloc[:required_count]
                    df_mapped = pd.concat([df_mapped[df_mapped['label'] != 'NORMAL'], normal_df])

                df_mapped['label'] = df_mapped['label_' + task]
                df_mapped['data'] = df_mapped['data_original']

                lbl_itos = df_mapped['label_' + task].unique()

                df_mapped['label'] = df_mapped['label'].apply(lambda x: np.array([float(x)]))
                
                df_train, df_val = [], []
                for cls in lbl_itos:
                    logger.info('Processing cls ' + str(cls))
                    class_df = df_mapped[df_mapped['label_' + task] == cls]
                    class_names = class_df['data_original'].apply(lambda x: x.split('@')[0])
                    class_unique_names = class_names.unique()
                    class_names_train, class_names_test = train_test_split(class_unique_names, test_size=0.2, random_state=666)
                    logger.info('Processing cls ' +  str(cls) + ' train')
                    class_names_train = set(class_names_train)
                    index = []
                    for name in class_names: index.append(True if name in class_names_train else False)
                    df_train.append(class_df[index])
                    logger.info('Processing cls ' +  str(cls) + ' test')
                    class_names_test = set(class_names_test)
                    index = []
                    for name in class_names: index.append(True if name in class_names_test else False)
                    df_val.append(class_df[index])
                df_train = pd.concat(df_train)
                df_val = pd.concat(df_val)
                logger.info('Processing ' + self.hparams.finetune_dataset + ' over')
                if(self.hparams.finetune):
                    df_test = df_val
            
            elif self.hparams.finetune_dataset.startswith("dandaolian"):
                logger.info('Processing ' + self.hparams.finetune_dataset)
                task = self.hparams.finetune_dataset.split('_')[-1]
                # df_mapped = df_mapped[df_mapped['label_' + task] >= 0]

                # df_mapped = pd.concat([df_mapped.iloc[0:500000], df_mapped.iloc[-500000:]])

                max_count_other_classes = df_mapped[df_mapped['label'] != 'NORMAL']['label'].value_counts().max()

                # 决定保留特定类数量的倍数
                n = 1

                if n > 0:
                    normal_df = df_mapped[df_mapped['label'] == 'NORMAL']
                    required_count = max_count_other_classes * n
                    logger.info('Remain count of NORMAL samples: ' + str(required_count))
                    if len(normal_df) > required_count:
                        normal_df = normal_df.iloc[:required_count]
                    df_mapped = pd.concat([df_mapped[df_mapped['label'] != 'NORMAL'], normal_df])

                df_mapped['label'] = df_mapped['label_' + task]

                lbl_itos = df_mapped['label_' + task].unique()

                df_mapped['label'] = df_mapped['label'].apply(lambda x: np.array([float(x)]))
                
                df_train, df_val = [], []
                for cls in lbl_itos:
                    logger.info('Processing cls ' + str(cls))
                    class_df = df_mapped[df_mapped['label_' + task] == cls]
                    class_names = class_df['data_original'].apply(lambda x: x.split('@')[0])
                    class_unique_names = class_names.unique()
                    class_names_train, class_names_test = train_test_split(class_unique_names, test_size=0.2, random_state=666)
                    logger.info('Processing cls ' +  str(cls) + ' train')
                    class_names_train = set(class_names_train)
                    index = []
                    for name in class_names: index.append(True if name in class_names_train else False)
                    df_train.append(class_df[index])
                    logger.info('Processing cls ' +  str(cls) + ' test')
                    class_names_test = set(class_names_test)
                    index = []
                    for name in class_names: index.append(True if name in class_names_test else False)
                    df_val.append(class_df[index])
                df_train = pd.concat(df_train)
                df_val = pd.concat(df_val)
                logger.info('Processing ' + self.hparams.finetune_dataset + ' over')
                if(self.hparams.finetune):
                    df_test = df_val
            
            else:
                max_fold_id = df_mapped.strat_fold.max() #unfortunately 1-based for PTB-XL; sometimes 100 (Ribeiro)
                df_train = df_mapped[df_mapped.strat_fold<(max_fold_id-1 if self.hparams.finetune else max_fold_id)]
                # df_train = df_mapped[df_mapped.strat_fold==(1 if self.hparams.finetune else max_fold_id)]
                df_val = df_mapped[df_mapped.strat_fold==(max_fold_id-1 if self.hparams.finetune else max_fold_id)]
                if(self.hparams.finetune):
                    df_test = df_mapped[df_mapped.strat_fold==max_fold_id]
            
            self.lbl_itos = lbl_itos
            tfms_ptb_xl_cpc = ToTensor() if self.hparams.normalize is False else transforms.Compose([Normalize(mean,std),ToTensor()])         
            
            train_datasets.append(TimeseriesDatasetCrops(df_train,self.hparams.input_size,num_classes=len(lbl_itos),data_folder=data_folder,chunk_length=chunk_length_train,min_chunk_length=self.hparams.input_size, stride=stride_train,transforms=tfms_ptb_xl_cpc,annotation=False,col_lbl ="label" if self.hparams.finetune else None,memmap_filename=target_folder/("memmap.npy") if not self.hparams.finetune_dataset.startswith("xinjikang") else None,multi_modal_dim=self.hparams.modal_dim if self.hparams.multi_modal else None))
            val_datasets.append(TimeseriesDatasetCrops(df_val,self.hparams.input_size,num_classes=len(lbl_itos),data_folder=data_folder,chunk_length=chunk_length_valtest,min_chunk_length=self.hparams.input_size, stride=stride_valtest,transforms=tfms_ptb_xl_cpc,annotation=False,col_lbl ="label" if self.hparams.finetune else None,memmap_filename=target_folder/("memmap.npy") if not self.hparams.finetune_dataset.startswith("xinjikang") else None,multi_modal_dim=self.hparams.modal_dim if self.hparams.multi_modal else None))
            if(self.hparams.finetune):
                test_datasets.append(TimeseriesDatasetCrops(df_test,self.hparams.input_size,num_classes=len(lbl_itos),data_folder=data_folder,chunk_length=chunk_length_valtest,min_chunk_length=self.hparams.input_size, stride=stride_valtest,transforms=tfms_ptb_xl_cpc,annotation=False,col_lbl ="label",memmap_filename=target_folder/("memmap.npy") if not self.hparams.finetune_dataset.startswith("xinjikang") else None,multi_modal_dim=self.hparams.modal_dim if self.hparams.multi_modal else None))
            
            print("\n",target_folder)
            print("train dataset:",len(train_datasets[-1]),"samples")
            print("val dataset:",len(val_datasets[-1]),"samples")
            if(self.hparams.finetune):
                print("test dataset:",len(test_datasets[-1]),"samples")

        if(len(train_datasets)>1): #multiple data folders
            print("\nCombined:")
            self.train_dataset = ConcatDataset(train_datasets)
            self.val_dataset = ConcatDataset(val_datasets)
            print("train dataset:",len(self.train_dataset),"samples")
            print("val dataset:",len(self.val_dataset),"samples")
            if(self.hparams.finetune):
                self.test_dataset = ConcatDataset(test_datasets)
                print("test dataset:",len(self.test_dataset),"samples")
        else: #just a single data folder
            self.train_dataset = train_datasets[0]
            self.val_dataset = val_datasets[0]
            if(self.hparams.finetune):
                self.test_dataset = test_datasets[0]

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, train_batch, batch_idx):
        return self._step(train_batch, batch_idx, True)
    
    def validation_step(self, val_batch, batch_idx, datalodaer_idx=0):
        return self._step(val_batch, batch_idx, False)
    
    def _step(self, data_batch, batch_idx, train):
        preds = self.forward(data_batch[0])
        loss = self.criterion(preds, data_batch[1])
        self.log("train_loss" if train else "val_loss", loss)
        return {'loss':loss, 'preds':preds.detach(), 'targs':data_batch[1]}
    
    def validation_epoch_end(self, outputs_all):
        res, res_agg = None, None
        print("Epoch %02d" % self.current_epoch, end=' ')
        for dataloader_idx,outputs in enumerate(outputs_all): #multiple val dataloaders
            print("Loader" + str(dataloader_idx), end=' ')
            preds_all = torch.cat([x['preds'] for x in outputs])
            targs_all = torch.cat([x['targs'] for x in outputs])
            if(self.hparams.finetune_dataset=="thew" or self.hparams.finetune_dataset in ['fangchan_all', 'fangchan_combine']):
                preds_all = F.softmax(preds_all, dim=-1)
                targs_all = torch.eye(len(self.lbl_itos))[targs_all].to(preds_all.device) 
                acctype = 'single' # 最终输出只有一个类别为True
            else:
                preds_all = torch.sigmoid(preds_all)
                acctype = 'multi' # 最终输出有多个类别为True
            preds_all = preds_all.cpu().numpy()
            targs_all = targs_all.cpu().numpy()
            #instance level score
            res = eval_scores(targs_all, preds_all, classes=self.lbl_itos, acctype=acctype)
            
            idmap = self.val_dataset.get_id_mapping()
            if idmap.max() + 1 != len(idmap):
                preds_all_agg,targs_all_agg = aggregate_predictions(preds_all,targs_all,idmap,aggregate_fn=np.mean)
                res_agg = eval_scores(targs_all_agg, preds_all_agg, classes=self.lbl_itos, acctype=acctype)
                self.log_dict({"macro_auc_agg"+str(dataloader_idx):res_agg["label_AUC"]["macro"], "macro_auc_noagg"+str(dataloader_idx):res["label_AUC"]["macro"]})
                print("Agg [MacroAUC {:.4f}][Acc {:.4f}]".format(res_agg["label_AUC"]["macro"], res_agg["acc"]), end=' ')
            print("NoAgg [MacroAUC {:.4f}][Acc {:.4f}]".format(res["label_AUC"]["macro"], res["acc"]), end=' ')
            self.log("macro_auc"+str(dataloader_idx),res["label_AUC"]["macro"])
        if self.best_metric is None:
            self.best_metric = res_agg if res_agg else res
        else:
            if res_agg:
                if res_agg["label_AUC"]["macro"] > self.best_metric["label_AUC"]["macro"]:
                    self.best_metric = res_agg
            else:
                if res["label_AUC"]["macro"] > self.best_metric["label_AUC"]["macro"]:
                    self.best_metric = res
            
        print()

    def custom_collate_fn(self, batch):
        Lens = [item[1].shape[0] for item in batch]
        maxLen = max(Lens)
        newbatch = []
        for item in batch:
            if item[1].shape[0] < maxLen:
                new_modal_data = torch.cat([item[1], torch.zeros((maxLen - item[1].shape[0], item[1].shape[1]))], axis=0)
                newbatch.append((item[0], new_modal_data, item[1].shape[0], item[2]))
            else:
                newbatch.append((item[0], item[1], item[1].shape[0], item[2]))
        ret = default_collate(newbatch)
        return ((ret[0], ret[1], ret[2]), ret[3])
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, num_workers=4, shuffle=True, drop_last=True, pin_memory=True, collate_fn=self.custom_collate_fn)
        
    def val_dataloader(self):
        if(self.hparams.finetune):#multiple val dataloaders
            return [DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, num_workers=4, pin_memory=True, collate_fn=self.custom_collate_fn),
                    DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, num_workers=4, pin_memory=True, collate_fn=self.custom_collate_fn)]
        else:
            return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, num_workers=4, collate_fn=self.custom_collate_fn)


def add_model_specific_args(parser):
    parser.add_argument("--input-channels", type=int, default=12)
    parser.add_argument("--normalize", action='store_true', help='Normalize input using PTB-XL stats')
    parser.add_argument('--mlp', action='store_true', help="False: original CPC True: as in SimCLR")
    parser.add_argument('--bias', action='store_true', help="original CPC: no bias")
    parser.add_argument("--n-hidden", type=int, default=512)
    parser.add_argument("--gru", action="store_true")
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--steps-predicted", dest="steps_predicted", type=int, default=12)
    parser.add_argument("--n-false-negatives", dest="n_false_negatives", type=int, default=128)
    parser.add_argument("--skip-encoder", action="store_true", help="disable the convolutional encoder i.e. just RNN; for testing")
    parser.add_argument("--fc-encoder", action="store_true", help="use a fully connected encoder (as opposed to an encoder with strided convs)")
    parser.add_argument("--negatives-from-same-seq-only", action="store_true", help="only draw false negatives from same sequence (as opposed to drawing from everywhere)")
    parser.add_argument("--no-bn-encoder", action="store_true", help="switch off batch normalization in encoder")
    parser.add_argument("--dropout-head", type=float, default=0.5)
    parser.add_argument("--train-head-only", action="store_true", help="freeze everything except classification head (note: --linear-eval defaults to no hidden layer in classification head)")
    parser.add_argument("--lin-ftrs-head", type=str, default="[512]", help="hidden layers in the classification head")
    parser.add_argument('--no-bn-head', action='store_true', help="use no batch normalization in classification head")
    parser.add_argument('--multi-modal', action='store_true', help="use text embedding as multi-modal data")
    parser.add_argument("--modal-dim", type=int, default=1792)
    return parser

def add_default_args():
    parser = argparse.ArgumentParser(description='PyTorch Lightning Baseline Training')
    parser.add_argument('--meta', metavar='DIR',type=str,
                        help='path(s) to meta',action='append')
    parser.add_argument('--data', metavar='DIR',type=str,
                        help='path(s) to dataset',action='append')
    parser.add_argument('--model', default='fcn_wang', type=str,
                        help='baseline model')
    parser.add_argument('--expnum', default=1, type=int, metavar='N',
                        help='number of total experiments to run')
    parser.add_argument('--epochs', default=30, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--batch-size', default=64, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float,
                        metavar='W', help='weight decay (default: 0.)',
                        dest='weight_decay')

    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

    parser.add_argument('--pretrained', default='', type=str, metavar='PATH',
                    help='path to pretrained checkpoint (default: none)')
    parser.add_argument('--optimizer', default='adam', help='sgd/adam')#was sgd
    parser.add_argument('--output-path', default='.', type=str,dest="output_path",
                        help='output path')
    parser.add_argument('--metadata', default='', type=str,
                        help='metadata for output')
    
    parser.add_argument("--gpus", type=int, default=1, help="number of gpus")
    parser.add_argument("--num-nodes", dest="num_nodes", type=int, default=1, help="number of compute nodes")
    parser.add_argument("--precision", type=int, default=16, help="16/32")
    parser.add_argument("--distributed-backend", dest="distributed_backend", type=str, default=None, help="None/ddp")
    parser.add_argument("--accumulate", type=int, default=1, help="accumulate grad batches (total-bs=accumulate-batches*bs)")
        
    parser.add_argument("--input-size", dest="input_size", type=int, default=16000)
    
    parser.add_argument("--finetune", action="store_true", help="finetuning (downstream classification task)",  default=False )
    parser.add_argument("--linear-eval", action="store_true", help="linear evaluation instead of full finetuning",  default=False )
    
    parser.add_argument(
        "--finetune-dataset",
        type=str,
        help="thew/ptbxl_super/ptbxl_all",
        default="thew"
    )
    
    parser.add_argument(
        "--discriminative-lr-factor",
        type=float,
        help="factor by which the lr decreases per layer group during finetuning",
        default=0.1
    )
    
    
    parser.add_argument(
        "--lr-find",
        action="store_true",
        help="run lr finder before training run",
        default=False
    )
    
    return parser
             
###################################################################################################
#MAIN
###################################################################################################
if __name__ == '__main__':
    parser = add_default_args()
    parser = add_model_specific_args(parser)
    hparams = parser.parse_args()
    hparams.executable = "cpc"

    if not os.path.exists(hparams.output_path):
        os.makedirs(hparams.output_path)
    
    tblogger = TensorBoardLogger(
        save_dir=hparams.output_path,
        #version="",#hparams.metadata.split(":")[0],
        name="")
    print("Output directory:",tblogger.log_dir)    
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(tblogger.log_dir,"best_model"),#hparams.output_path
        save_top_k=1,
		save_last=True,
        verbose=True,
        monitor='macro_auc0' if hparams.finetune else 'val_loss',#val_loss/dataloader_idx_0
        mode='max' if hparams.finetune else 'min')
    lr_monitor = LearningRateMonitor()
    
    aucs = []
    accs = []
    for i in range(hparams.expnum):
        logger.info('Exp' + str(i))
        model = LightningBaseline(hparams)
        
        if(hparams.pretrained!=""):
            print("Loading pretrained weights from",hparams.pretrained)
            model.load_weights_from_checkpoint(hparams.pretrained)

        trainer = pl.Trainer(
            #overfit_batches=0.01,
            auto_lr_find = hparams.lr_find,
            accumulate_grad_batches=hparams.accumulate,
            max_epochs=hparams.epochs,
            min_epochs=hparams.epochs,
            
            default_root_dir=hparams.output_path,
            
            num_sanity_val_steps=0,
            
            logger=tblogger,
            checkpoint_callback=False,
            callbacks = [],#lr_monitor],
            benchmark=True,
        
            gpus=hparams.gpus,
            num_nodes=hparams.num_nodes,
            precision=hparams.precision,
            distributed_backend=hparams.distributed_backend,
            
            progress_bar_refresh_rate=0,
            weights_summary='top',
            resume_from_checkpoint= None if hparams.resume=="" else hparams.resume)
        
        if(hparams.lr_find):#lr find
            trainer.tune(model)
            
        trainer.fit(model)
        
        logger.info("Best [MacroAUC {:.4f}][Acc {:.4f}]".format(model.best_metric["label_AUC"]["macro"], model.best_metric["acc"]))
        aucs.append(model.best_metric["label_AUC"]["macro"])
        accs.append(model.best_metric["acc"])
    
    if hparams.expnum > 1:
        print("MacroAUC Mean {:.4f} Min {:.4f} Max {:.4f} | ".format(np.mean(aucs), np.min(aucs), np.max(aucs)), end=' ')
        for auc in aucs:
            print('{:.4f}'.format(auc), end=' ')
        print()
        print("Acc Mean {:.4f} Min {:.4f} Max {:.4f} | ".format(np.mean(accs), np.min(accs), np.max(accs)), end=' ')
        for acc in accs:
            print('{:.4f}'.format(acc), end=' ')
        print()

