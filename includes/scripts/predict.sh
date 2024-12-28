#!/bin/bash

# config.json 파일에서 설정 값 불러오기
CONFIG_JSON=$(cat config.json)

REGISTER_NAME=$(echo "$CONFIG_JSON" | jq -r '.REGISTER_NAME')
IMAGE_PATH=$(echo "$CONFIG_JSON" | jq -r '.IMAGE_PATH')

# 이미지 파일이 존재하는지 확인
if [ ! -f "$IMAGE_PATH" ]; then
    echo "Error: Image file '$IMAGE_PATH' not found!"
    exit 1
else
    echo "Image file '$IMAGE_PATH' found."
fi

# 이미지를 base64로 인코딩
IMAGE_BASE64=$(base64 -i "$IMAGE_PATH")

# 디버깅: Base64로 인코딩된 이미지 일부 출력
# echo "Base64 encoded image (first 100 characters):"
# echo "$IMAGE_BASE64" | head -n 1

# JSON 형식으로 데이터 구성
DATA=$(cat <<EOF
{
    "image": "$IMAGE_BASE64",
    "register_name": "$REGISTER_NAME"
}
EOF
)

# 디버깅: 생성된 JSON 출력
# echo "Generated JSON data:"
# echo "$DATA"

# POST 요청 보내기
response=$(curl -s -X POST http://localhost:5001/predict \
    -H "Content-Type: application/json" \
    -d "$DATA")

# 서버 응답 출력
echo "Response: $response"
