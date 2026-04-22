# export CUDA_VISIBLE_DEVICES=0
python main_cpc_lightning.py \
    --output-path /DATA/DATANAS1/hrz/azhospital/SeqCls/iKang/runs/cpc/all_xianweihua \
    --meta /DATA/DATANAS1/hrz/azhospital/SeqCls/iKang/ecg_data_processed/xianweihua \
    --data /DATA/DATANAS1/hrz/azhospital/SeqCls/iKang/ecg_data_processed/xianweihua \
    --expnum 1 \
    --normalize \
    --input-size 250 \
    --epochs 50 \
    --lr 0.001 \
    --batch-size 128 \
    --fc-encoder \
    --finetune --pretrained /DATA/DATANAS1/hrz/azhospital/SeqCls/iKang/checkpoints/cpc_on_all.pt --finetune-dataset xianweihua_sub --train-head-only