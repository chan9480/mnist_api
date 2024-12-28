import os
import mlflow
import base64

from flask import Flask, request, jsonify
from includes.source.training import Trainer
from includes.source.registration import ModelRegistrar
from includes.source.prediction import Predictor

app = Flask(__name__)
mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow-server:5000'))

@app.route('/train', methods=['POST'])
def train():
    data = request.get_json()

    # 요청에서 하이퍼파라미터 값들을 추출
    learning_rate = data.get('learning_rate', 0.001)
    batch_size = data.get('batch_size', 64)
    epochs = data.get('epochs', 10)
    run_name = data.get('run_name', 'default_run')  # run_name을 받아옵니다
    experiment_name = data.get('experiment_name', 'Default')  # experiment_name을 받아옵니다
    val_split = data.get('val_split', 0.2)
    loss_function = data.get('loss_function', 'CrossEntropyLoss')
    optimizer_type = data.get('optimizer_type', 'Adam')

    # Trainer 객체 생성
    trainer = Trainer(learning_rate, batch_size, epochs, run_name, experiment_name, val_split,
                      loss_function, optimizer_type)
    
    # 모델 훈련
    result = trainer.train_model()

    # 결과를 응답으로 반환
    return jsonify({
        "message": "Training complete",
        "model_path": result['model_path'],
        "final_epoch": result['final_epoch'],
        "train_loss": result['train_loss'],
        "val_loss": result['val_loss'],
        "val_accuracy": result['val_accuracy'],
        "learning_rate": result['learning_rate'],
        "batch_size": result['batch_size'],
        "loss_function": result['loss_function'],
        "optimizer": result['optimizer']
    })

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    experiment_name = data.get('experiment_name')
    run_name = data.get('run_name')
    register_name = data.get('register_name')

    if not experiment_name or not run_name:
        return jsonify({"error": "Experiment Name and Run Name are required"}), 400

    try:
        registrar = ModelRegistrar(experiment_name, run_name, register_name)
        model_info = registrar.register_model()

        # model_info에 포함된 값을 클라이언트에게 반환
        return jsonify({
            "message": "Model registered successfully",
            "model_version": model_info["model_version"],
            "model_name": model_info["model_name"],
            "onnx_model_uri": model_info["onnx_model_uri"],
            "onnx_model_path": model_info["onnx_model_path"],
            "model_structure": model_info["model_structure"]
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image_base64 = data.get('image')
    register_name = data.get('register_name', 'mnist_model')  # Default to 'mnist_model' if not provided
    
    if not image_base64:
        return jsonify({"error": "No image provided"}), 400

    # Decode the base64 encoded image data
    image_data = base64.b64decode(image_base64)
    
    # Initialize the Predictor with the provided or default register name
    predictor = Predictor(register_name)
    
    # Get the prediction and confidence
    predicted_class, confidence = predictor.predict_image(image_data)

    # Return the prediction and confidence as JSON
    return jsonify({
        "prediction": predicted_class,
        "confidence": confidence
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
