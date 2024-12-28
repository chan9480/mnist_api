import mlflow
import mlflow.pytorch
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torch import nn
from sklearn.model_selection import train_test_split
from includes.source.model import CNN  # CNN 모델 import


class Trainer:
    def __init__(self, learning_rate, batch_size, epochs, run_name=None, experiment_name=None, val_split=0.2):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.run_name = run_name
        self.val_split = val_split  # 검증 데이터 비율
        self.experiment_name = experiment_name  # 실험 이름

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def _get_experiment_id(self):
        """주어진 experiment_name으로 experiment_id를 반환하고, 없으면 생성"""
        # experiment_name이 주어지면 해당 이름의 실험을 찾고, 없으면 새로 생성
        experiment = mlflow.get_experiment_by_name(self.experiment_name)

        if experiment is None:
            # 실험이 존재하지 않으면 새로 생성
            experiment_id = mlflow.create_experiment(self.experiment_name)
        else:
            # 실험이 존재하면 그 ID를 반환
            experiment_id = experiment.experiment_id
        
        return experiment_id

    def train_model(self):
        # 데이터 로드
        train_data = datasets.MNIST(root='./includes/data', train=True, download=True, transform=self.transform)

        # 훈련 데이터와 검증 데이터 분리
        train_indices, val_indices = train_test_split(range(len(train_data)), test_size=self.val_split)
        train_subset = Subset(train_data, train_indices)
        val_subset = Subset(train_data, val_indices)

        train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=self.batch_size, shuffle=False)

        model = CNN()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

        # 실험 ID 얻기 (없으면 새로 생성)
        experiment_id = self._get_experiment_id()

        # MLflow 실험 시작 (experiment_id가 주어지면 해당 실험에서 실행)
        with mlflow.start_run(run_name=self.run_name, experiment_id=experiment_id):
            mlflow.log_param("learning_rate", self.learning_rate)
            mlflow.log_param("batch_size", self.batch_size)
            mlflow.log_param("epochs", self.epochs)

            for epoch in range(self.epochs):
                model.train()
                running_loss = 0.0
                correct = 0
                total = 0

                # 훈련
                for inputs, labels in train_loader:
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                mlflow.log_metric("train_loss", running_loss / len(train_loader))
                mlflow.log_metric("train_accuracy", 100 * correct / total)

                # 검증
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()

                mlflow.log_metric("val_loss", val_loss / len(val_loader))
                mlflow.log_metric("val_accuracy", 100 * val_correct / val_total)

                print(f"Epoch [{epoch+1}/{self.epochs}], "
                      f"Train Loss: {running_loss/len(train_loader):.4f}, Train Accuracy: {100 * correct / total:.2f}%, "
                      f"Val Loss: {val_loss / len(val_loader):.4f}, Val Accuracy: {100 * val_correct / val_total:.2f}%")

            model_path = './includes/models/pth/mnist_model.pth'
            torch.save(model.state_dict(), model_path)
            mlflow.pytorch.log_model(model, "model")
            mlflow.log_artifact(model_path)

        return model_path
