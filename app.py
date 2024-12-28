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

    learning_rate = data.get('learning_rate', 0.001)
    batch_size = data.get('batch_size', 64)
    epochs = data.get('epochs', 10)
    run_name = data.get('run_name', 'default_run')  # run_name을 받아옵니다
    experiment_id = data.get('experiment_name', 'Default')

    trainer = Trainer(learning_rate, batch_size, epochs, run_name, experiment_id)  # run_name 전달
    model_path = trainer.train_model()

    return jsonify({     
        "message": "Training complete",
        "model_path": model_path
    })

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    experiment_name = data.get('experiment_name')
    run_name = data.get('run_name')
    register_name = data.get('register_name')

    if not experiment_name or not run_name:
        return jsonify({"error": "Experiment ID and Run Name are required"}), 400

    try:
        registrar = ModelRegistrar(experiment_name, run_name, register_name)
        model_version = registrar.register_model()

        return jsonify({
            "message": "Model registered successfully",
            "mlflow_model_version": model_version
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image_base64 = data.get('image')
    register_name = data.get('register_name')
    
    if not image_base64:
        return jsonify({"error": "No image provided"}), 400

    image_data = base64.b64decode(image_base64)
    
    predictor = Predictor(register_name)
    prediction = predictor.predict_image(image_data)

    return jsonify({
        "prediction": prediction
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
