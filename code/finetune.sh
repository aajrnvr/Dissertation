python ./finetune.py \
  --task_name readmission \
  --readmission_mode discharge \
  --do_train \
  --do_eval \
  --data_dir ./fewshotdata_for_finetune/few_shot_data \
  --bert_model ./model/pretraining/pretraining/clinical-pubmed-bert-base-512/for_prompt \
  --max_seq_length 512 \
  --train_batch_size 4 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir result_all_finetune/finetune \
  --seed 1 \
  --date 2_21 \
  --tune_plm \