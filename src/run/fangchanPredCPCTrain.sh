# export CUDA_VISIBLE_DEVICES=0
python main_cpc_lightning.py \
    --output-path /home/user/code/SeqCls/runs/cpc/all_fangchan \
    --meta /home/user/code/SeqCls/ecg_data_processed/fangchan_fs100 \
    --data /home/user/code/SeqCls/ecg_data_processed/fangchan_fs100 \
    --expnum 10 \
    --normalize \
    --input-size 1000 \
    --epochs 100 \
    --lr 0.0001 \
    --batch-size 128 \
    --fc-encoder \
    --multi-modal --modal-dim 3584 \
    --finetune --finetune-dataset fangchan_pred