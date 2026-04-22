export CUDA_VISIBLE_DEVICES=0
python main_cpc_lightning.py \
    --output-path ../runs/cpc/all_fangchan \
    --data ../ecg_data_processed/fangchan_fs100 \
    --normalize \
    --input-size 250 \
    --epochs 100 \
    --lr 0.0001 \
    --batch-size 128 \
    --fc-encoder \
    --finetune --pretrained ../checkpoints/cpc_on_all.pt --finetune-dataset fangchan_pred