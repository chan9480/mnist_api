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
    def __init__(self, learning_rate, batch_size, epochs, run_name=None, experiment_name=None, val_split=0.2,
                 loss_function_str="CrossEntropyLoss", optimizer_type_str="Adam"):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.run_name = run_name
        self.val_split = val_split
        self.experiment_name = experiment_name

        # 옵티마이저와 손실 함수의 문자열을 받아서 적절한 객체로 설정
        self.loss_function_str = loss_function_str
        self.optimizer_type_str = optimizer_type_str
        
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def _get_experiment_id(self):
        """주어진 experiment_name으로 experiment_id를 반환하고, 없으면 생성"""
        experiment = mlflow.get_experiment_by_name(self.experiment_name)

        if experiment is None:
            experiment_id = mlflow.create_experiment(self.experiment_name)
        else:
            experiment_id = experiment.experiment_id
        
        return experiment_id

    def get_loss_function(self):
        """손실 함수 문자열에 따라 손실 함수 객체 반환"""
        if self.loss_function_str == "CrossEntropyLoss":
            return nn.CrossEntropyLoss()
        elif self.loss_function_str == "MSELoss":
            return nn.MSELoss()
        elif self.loss_function_str == "BCEWithLogitsLoss":
            return nn.BCEWithLogitsLoss()
        elif self.loss_function_str == "SmoothL1Loss":
            return nn.SmoothL1Loss()
        elif self.loss_function_str == "L1Loss":
            return nn.L1Loss()
        else:
            raise ValueError(f"지원되지 않는 손실 함수: {self.loss_function_str}")

    def get_optimizer(self, model):
        """옵티마이저 문자열에 따라 옵티마이저 객체 반환"""
        if self.optimizer_type_str == "Adam":
            return optim.Adam(model.parameters(), lr=self.learning_rate)
        elif self.optimizer_type_str == "AdamW":
            return optim.AdamW(model.parameters(), lr=self.learning_rate)
        elif self.optimizer_type_str == "SGD":
            return optim.SGD(model.parameters(), lr=self.learning_rate, momentum=0.9)
        elif self.optimizer_type_str == "RMSprop":
            return optim.RMSprop(model.parameters(), lr=self.learning_rate)
        elif self.optimizer_type_str == "Adagrad":
            return optim.Adagrad(model.parameters(), lr=self.learning_rate)
        elif self.optimizer_type_str == "Adadelta":
            return optim.Adadelta(model.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"지원되지 않는 옵티마이저: {self.optimizer_type_str}")

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

        # 손실 함수와 옵티마이저 설정
        criterion = self.get_loss_function()
        optimizer = self.get_optimizer(model)

        experiment_id = self._get_experiment_id()

        with mlflow.start_run(run_name=self.run_name, experiment_id=experiment_id):
            mlflow.log_param("learning_rate", self.learning_rate)
            mlflow.log_param("batch_size", self.batch_size)
            mlflow.log_param("epochs", self.epochs)
            mlflow.log_param("loss_function", self.loss_function_str)
            mlflow.log_param("optimizer", self.optimizer_type_str)

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

        # 최종 epoch 및 훈련/검증 결과 반환
        return {
            "model_path": model_path,
            "final_epoch": self.epochs,
            "train_loss": running_loss / len(train_loader),
            "val_loss": val_loss / len(val_loader),
            "val_accuracy": 100 * val_correct / val_total,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "loss_function": self.loss_function_str,
            "optimizer": self.optimizer_type_str
        }
