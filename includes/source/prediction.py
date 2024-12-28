import torch
import mlflow
import mlflow.pytorch
from io import BytesIO
from PIL import Image
import onnx
import onnxruntime as ort
from includes.source.utils import transform
from torchvision import transforms

class Predictor:
    def __init__(self, register_name='mnist_model'):
        self.model_name = register_name

    def load_model(self):
        # MLflow에서 가장 최근 버전의 ONNX 모델을 불러오기
        model_uri = f"models:/{self.model_name}/latest"
        print(model_uri)
        
        # ONNX 모델 파일 경로 얻기
        local_model_path = mlflow.artifacts.download_artifacts(model_uri=model_uri)
        print(local_model_path)
        onnx_model_path = f"{local_model_path}/model.onnx"
        
        # ONNX 모델 로드
        onnx_model = onnx.load(onnx_model_path)
        
        # ONNX Runtime을 이용하여 세션을 생성
        ort_session = ort.InferenceSession(onnx_model_path)
        
        return ort_session

    def predict_image(self, image_data):
        model = self.load_model()

        # 이미지를 BytesIO로 열고, Grayscale로 변환
        img = Image.open(BytesIO(image_data)).convert('L')

        # 이미지 크기 변경
        img = img.resize((28, 28))

        # 이미지 변환
        img = transform(img).unsqueeze(0)  # 배치 차원 추가

        # 모델 예측
        with torch.no_grad():
            output = model(img)
            _, predicted = torch.max(output.data, 1)
            return predicted.item()
