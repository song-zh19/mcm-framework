export CUDA_VISIBLE_DEVICES=0
python main_cpc_lightning.py \
    --output-path ../runs/cpc/all_dandaolian \
    --meta ../ecg_data_processed/dandaolian_fs64 \
    --data ../ \
    --input-channels 1 \
    --input-size 1920 \
    --epochs 50 \
    --lr 0.0001 \
    --batch-size 128 \
    --fc-encoder \
    --finetune --finetune-dataset dandaolian_all