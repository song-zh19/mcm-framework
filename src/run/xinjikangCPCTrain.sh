export CUDA_VISIBLE_DEVICES=1
python main_cpc_lightning.py \
    --output-path ../runs/cpc/all_xinjikang \
    --meta /home/hrz/ecg_data_processed/xinjikang_fs64 \
    --data /DATA/DATANAS1/hrz/azhospital/SeqCls/iKang \
    --input-channels 1 \
    --input-size 1920 \
    --epochs 100 \
    --lr 0.00002 \
    --batch-size 32 \
    --fc-encoder \
    --finetune --finetune-dataset xinjikang_all