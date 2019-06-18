#!/usr/bin/env bash

yudctl run \
        -t bert_predict_intent \
        -g 1 \
        -p /data/share/liuchang/bert/  \
        -i registry.cn-hangzhou.aliyuncs.com/eigenlab/yudexcutor:tf1.12 \
        -r requirements.txt -- scripts/predict_intent.sh