for num in 8 16 32 64 128 256
  do
    for seed in 100 121 146 11 300 13
      do
        python ./generate_data_for_finetune.py \
          --num_examples_per_label $num \
          --seed $seed \
        &  
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
          --in_loop_setting
      done
  done