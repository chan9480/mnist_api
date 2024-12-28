import mlflow
import torch
import torch.onnx
from mlflow import pytorch
from mlflow.tracking import MlflowClient
import os

class ModelRegistrar:
    def __init__(self, experiment_name, run_name, register_name):
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.register_name = register_name
        mlflow.set_experiment(self.experiment_name)

    def get_run_id_by_name(self):
        # 주어진 experiment_name과 run_name에 해당하는 run_id를 찾기
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if not experiment:
            raise ValueError(f"Experiment with name '{self.experiment_name}' not found.")

        # 특정 run_name을 기준으로 run을 가져옴
        client = MlflowClient()
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.mlflow.runName = '{self.run_name}'"
        )

        if not runs:
            raise ValueError(f"No run found with name '{self.run_name}' in experiment '{self.experiment_name}'.")

        return runs[0].info.run_id

    def register_model(self):
        # 특정 run_id를 얻기 위해 run_name을 사용
        run_id = self.get_run_id_by_name()

        # MLflow에서 이미 저장된 모델을 로드
        with mlflow.start_run(run_id=run_id):  # 특정 run_id를 사용하여 모델을 등록
            # MLflow에서 PyTorch 모델을 로드
            model = mlflow.pytorch.load_model(f"runs:/{run_id}/model")

            # 모델을 ONNX 형식으로 변환
            onnx_model_path = self._convert_to_onnx(model, run_id)

            # ONNX 모델을 MLflow 아티팩트로 업로드
            artifact_path = f"onnx_model"
            mlflow.log_artifact(onnx_model_path, artifact_path)

            # 모델을 레지스트리에 등록
            onnx_model_uri = f"runs:/{run_id}/{artifact_path}"
            result = mlflow.register_model(onnx_model_uri, self.register_name)
            model_version = result.version

        mlflow.end_run()

        return model_version

    def _convert_to_onnx(self, model, run_id):
        # 모델을 추론 모드로 설정
        model.eval()

        # 더미 입력 데이터를 생성 (예: MNIST의 경우 크기 [1, 1, 28, 28])
        dummy_input = torch.randn(1, 1, 28, 28)

        # ONNX 모델 경로 설정
        onnx_model_path = f"onnx_model.onnx"

        # ONNX로 모델을 변환
        torch.onnx.export(model, dummy_input, onnx_model_path, export_params=True)

        return onnx_model_path
