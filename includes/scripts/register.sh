#!/bin/bash

# config.json 파일에서 설정 값 불러오기
CONFIG_JSON=$(cat config.json)

REGISTER_NAME=$(echo "$CONFIG_JSON" | jq -r '.REGISTER_NAME')
EXPERIMENT_NAME=$(echo "$CONFIG_JSON" | jq -r '.EXPERIMENT_NAME')
RUN_NAME=$(echo "$CONFIG_JSON" | jq -r '.RUN_NAME')

# JSON 형식으로 데이터 구성
DATA=$(cat <<EOF
{
    "experiment_name": "$EXPERIMENT_NAME",
    "run_name": "$RUN_NAME",
    "register_name": "$REGISTER_NAME"
}
EOF
)

# POST 요청 보내기
response=$(curl -s -X POST http://localhost:5001/register \
    -H "Content-Type: application/json" \
    -d "$DATA")

# 서버 응답 출력
echo "Response: $response"
