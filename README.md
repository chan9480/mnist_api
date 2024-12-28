# MNIS classification 학습, 등록, 추론을 위한 API
client에서 원하는 정보(하이퍼파라미터, experiment name, run name)을 서버의 api에 POST  
-> experiment 결과, 모델정보, 예측결과 등을 request
# 사용 skill
PyTorch, MLFlow, Flask, docker-compose, onnx
# server (docker-compose containers)
1. flask-api : 3개의 endpoint를 갖는 api를 제공하는 flask 구성. (5000 포트 -> 로컬 5001 포트)
- train : 하이퍼파라미터를 receive. metrics와 artifacts 를 로컬 실험 추적서버에 logging. 완료되면 추적된 실험 항목에 대한 정보를 return.
- register : 훈련 실험에 대한 참조를 수신하고, 해당 모델을 onnx 형식으로 export 모델 레지스트리로 등록, 모델정보를 return
- predict : 이미지 파일을 받아 최신 등록된 모델을 사용하여 숫자를 예측하고 예측 결과를 return.
2. mlflow-server : mlflow ui를 제공하는 서버 (5000포트 -> 5002 포트)

서버 구성 시
```
docker-compose up
```
자동으로 docker pull을 해주는걸로 알고있지만 안된다면 ghcr.io/mlflow/mlflow:v2.3.0 이미지가 없다면 따로 다운로드 후 `docker-compose up`
```
docker pull ghcr.io/mlflow/mlflow:v2.3.0
```

# client
1. http://0.0.0.0:5002/ 로 MLFlow web 접근 가능  

2. `config.json`의 정보를 수정 한다.

- LOSS_FUNCTION 아래 중 선택.
  - CrossEntropyLoss
  - MSELoss
  - BCEWithLogitsLoss
  - SmoothL1Loss
  - L1Loss  

- OPTIMIZER_TYPE 아래 중 선택.
  - Adam
  - AdamW
  - SGD
  - RMSprop
  - Adagrad
  - Adadelta

```
{
    "EXPERIMENT_NAME": "chanwoo_test",
    "RUN_NAME": "mnist_training_run",
    "EPOCHS": 1,
    "BATCH_SIZE": 8,
    "LEARNING_RATE": 0.001,
    "IMAGE_PATH": "includes/data/test.png",
    "REGISTER_NAME": "mnist_model_test",
    "VAL_SPLIT":0.2,
    "LOSS_FUNCTION":"CrossEntropyLoss",
    "OPTIMIZER_TYPE":"Adam"
  }
```

3. sh파일을 실행한다. jq 가 설치되어있지 않다면 설치. (.sh 파일내에서 사용함)
```
yum install jq
```
### 학습시
```
sh includes/scripts/post_train.sh
```
Response 예시
```
Response: {"batch_size":8,"final_epoch":1,"learning_rate":0.001,"loss_function":"CrossEntropyLoss","message":"Training complete","model_path":"./includes/models/pth/mnist_model.pth","optimizer":"Adam","train_loss":0.10925493176472477,"val_accuracy":98.11666666666666,"val_loss":0.0647734050079501}
```
### register 시
```
sh includes/scripts/register.sh
```
Response 예시
```
Response: {"message":"Model registered successfully","model_name":"mnist_model_test","model_structure":"CNN(
  (conv1): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1))
  (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
  (fc1): Linear(in_features=9216, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=10, bias=True)
)","model_version":"1","onnx_model_path":"onnx_model.onnx","onnx_model_uri":"runs:/d8a72c072c3e4af1a620873de4dbb728/onnx_model"}
```
### 추론시
```
sh includes/scripts/predict.sh
```
Response 예시
```
Response: {"prediction":8}
```

# 테스트환경
- docker-compose : v2.29.2-desktop.2  
- docker 버전 : 27.2.0, build 3ab4256
- apple-silicon : m2 pro
# TODO 
- flask multi-threading 구현
- model 구조 개선
- onnx 변환시 성능 차이 확인
- early stopping 기능 추가