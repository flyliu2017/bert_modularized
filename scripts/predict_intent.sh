#!/usr/bin/env bash

python -m bin.run --task_name=intent \
                    --do_predict=true \
                    --data_dir=/data/share/liuchang/intent/ \
                    --vocab_file=/data/share/ludezheng/bert/chinese_L-12_H-768_A-12/vocab.txt \
                    --bert_config_file=/data/share/ludezheng/bert/chinese_L-12_H-768_A-12/bert_config.json \
                    --init_checkpoint=/data/share/ludezheng/bert/chinese_L-12_H-768_A-12/bert_model.ckpt \
                    --max_seq_length=500 \
                    --train_batch_size=8 \
                    --learning_rate=1e-5 \
                    --num_train_epochs=50 \
                    --output_dir /data/share/liuchang/intent/bert_model \
                    --decay_type exp \
                    --warmup_proportion=0.005 \
                    --decay_steps=100 --decay_rate=0.99