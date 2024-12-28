#!/bin/bash

# config.json 파일에서 설정 값 불러오기
CONFIG_JSON=$(cat config.json)

EXPERIMENT_NAME=$(echo "$CONFIG_JSON" | jq -r '.EXPERIMENT_NAME')
RUN_NAME=$(echo "$CONFIG_JSON" | jq -r '.RUN_NAME')
EPOCHS=$(echo "$CONFIG_JSON" | jq -r '.EPOCHS')
BATCH_SIZE=$(echo "$CONFIG_JSON" | jq -r '.BATCH_SIZE')
LEARNING_RATE=$(echo "$CONFIG_JSON" | jq -r '.LEARNING_RATE')

# POST 요청 보내기
response=$(curl -s -X POST http://localhost:5001/train \
    -H "Content-Type: application/json" \
    -d '{
        "learning_rate": '"$LEARNING_RATE"',
        "batch_size": '"$BATCH_SIZE"',
        "epochs": '"$EPOCHS"',
        "run_name": "'"$RUN_NAME"'",
        "experiment_name": "'"$EXPERIMENT_NAME"'"
    }')

echo "Response: $response"
