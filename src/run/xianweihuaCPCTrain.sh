# export CUDA_VISIBLE_DEVICES=0
python main_cpc_lightning.py \
    --output-path /DATA/DATANAS1/hrz/azhospital/SeqCls/iKang/runs/cpc/all_xianweihua \
    --meta /DATA/DATANAS1/hrz/azhospital/SeqCls/iKang/ecg_data_processed/xianweihua \
    --data /DATA/DATANAS1/hrz/azhospital/SeqCls/iKang/ecg_data_processed/xianweihua \
    --expnum 10 \
    --normalize \
    --input-size 250 \
    --epochs 100 \
    --lr 0.0001 \
    --batch-size 32 \
    --fc-encoder \
    --multi-modal --modal-dim 3584 \
    --finetune --finetune-dataset xianweihua_sub

# acge 1024
# qwen1.5 4096
# qwen2 3584
# --multi-modal --modal-dim 1024 \