export CUDA_VISIBLE_DEVICES=0
python main_cpc_lightning.py \
    --output-path ../runs/cpc/all_ptbxl \
    --data ../ecg_data_processed/ptb_xl_fs100 \
    --normalize \
    --input-size 250 \
    --epochs 100 \
    --lr 0.0001 \
    --batch-size 128 \
    --fc-encoder \
    --finetune --finetune-dataset ptbxl_all
