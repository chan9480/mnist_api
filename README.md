# TODO 
 - 그외 필요한 정보 API 리턴으로 추가. 
 - predict 시 onnxruntime으로 추론하도록
 - 모든 이미지 삭제후. 사용 가능한지
 - jq가 깔려있지 않을시 설치해야한다는 안내.
# MNIS classification 학습과 제공을 위한 API
## 모델 registry 및 실험 트래킹을 포함

  
# endpoint 는 세가지. 1. 훈련 2. register 3. predict

# 훈련 : 하이퍼파라미터를 receive. metrics와 artifacts 를 로컬 실험 추적서버에 logging. 완료되면 추적된 실험 항목에 대한 정보를 return.
# register : 등록: 훈련 실험에 대한 참조를 수신하고, 해당 모델을 onnx 형식으로 export 모델 레지스트리로 승격합니다. 
  # 등록하기 전에 일부 모델 거버넌스 프로세스가 포함될 수 있습니다.(어떤 특정 role이 있어도 된다는 뜻?) 등록된 모델에 대한 정보를 return
# 예측: 이미지 파일을 받아 최신 등록된 모델을 사용하여 숫자를 예측하고 예측 결과를 return.

# Deep learning: PyTorch, Lightning, TensorFlow, etc.
# Model Registry and Experiment Tracking: MLFlow, Neptune, etc.
# API: Flask
# README 를 작성하시오. 20분정도의 presentaion
# docker/docker-compose를 사용하시오.

# api. 로 정보를 받는 모듈 -> flask == main
# flask에서 실행시킬 모듈들 
   # 학습코드 : 로컬에서 실행. -> 정보 
   # logging 코드
   # 예측코드

# TODO : 훈련을 할때의 실험결과를 register에서 그대로 참조하도록 할 수 있어야 한다!?