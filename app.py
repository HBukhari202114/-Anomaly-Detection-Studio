import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from lightgbm import LGBMClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import category_encoders as ce
import matplotlib
matplotlib.use('Agg')
import signal
import sys
import uuid
import shutil
import traceback

app = Flask(__name__, template_folder='templates', static_folder='static', static_url_path='/static')
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024
CORS(app, resources={r"/*": {"origins": "*"}})
executor = ThreadPoolExecutor()

def build_autoencoder(input_dim):
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(64, activation='relu')(input_layer)
    encoded = Dense(32, activation='relu')(encoded)
    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return autoencoder

def build_deep_classifier(input_dim):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def run_model(features_scaled, model_type, contamination=None, labels=None, threshold_mode='contamination'):
    dynamic_contam = contamination is None
    contamination = contamination if contamination else 0.1

    if model_type == 'isolation_forest':
        iso_contam = contamination if threshold_mode == 'contamination' else 'auto'
        model = IsolationForest(contamination=iso_contam)
        model.fit(features_scaled)
        return np.where(model.predict(features_scaled) == 1, 0, 1)

    elif model_type == 'autoencoder':
        model = build_autoencoder(features_scaled.shape[1])
        model.fit(features_scaled, features_scaled, epochs=50, batch_size=32, verbose=0)
        reconstructions = model.predict(features_scaled)
        mse = np.mean(np.power(features_scaled - reconstructions, 2), axis=1)
        if threshold_mode == 'contamination':
            threshold = np.quantile(mse, 1 - contamination)
        else:
            threshold = np.mean(mse) + 2 * np.std(mse)
        return (mse > threshold).astype(int)

    elif model_type == 'random_forest':
        clf = RandomForestClassifier()
        clf.fit(features_scaled, labels)
        return clf.predict(features_scaled)

    elif model_type == 'lightgbm':
        clf = LGBMClassifier()
        clf.fit(features_scaled, labels)
        return clf.predict(features_scaled)

    elif model_type == 'deep_learning':
        model = build_deep_classifier(features_scaled.shape[1])
        model.fit(features_scaled, labels, epochs=50, batch_size=32, verbose=0)
        preds = model.predict(features_scaled).flatten()
        return (preds > 0.5).astype(int)

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def process_file(file, models, target_col, contamination, threshold_mode):
    try:
        df = pd.read_csv(file.stream) if file.filename.endswith('.csv') else pd.read_json(file.stream)

        labels = None
        if target_col and target_col in df.columns:
            labels = df[target_col].copy().astype(str)
            df = df.drop(columns=[target_col])

        if df.empty:
            return {'error': 'No features remaining after target column removal'}

        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()

        imputer = SimpleImputer(strategy='mean')
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

        encoded_parts = []
        for col in categorical_cols:
            nunique = df[col].nunique()
            if nunique <= 50:
                oh = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                enc = oh.fit_transform(df[[col]])
                enc_df = pd.DataFrame(enc, columns=oh.get_feature_names_out([col]))
                encoded_parts.append(enc_df)
            elif nunique <= 500:
                df[col] = LabelEncoder().fit_transform(df[col].astype(str))
                encoded_parts.append(df[[col]])
            else:
                he = ce.HashingEncoder(cols=[col], n_components=32)
                hash_df = he.fit_transform(df[[col]])
                encoded_parts.append(hash_df)

        df = df.drop(columns=categorical_cols)
        feature_df = pd.concat([df[numeric_cols]] + encoded_parts, axis=1)

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(feature_df)

        if labels is not None:
            le_full = LabelEncoder().fit(labels.astype(str))
            feature_df[target_col] = le_full.transform(labels.astype(str))

        preprocessed_filename = f"preprocessed_{uuid.uuid4().hex}.csv"
        preprocessed_path = os.path.join("static", "temp", preprocessed_filename)
        os.makedirs(os.path.dirname(preprocessed_path), exist_ok=True)
        feature_df.to_csv(preprocessed_path, index=False)

        results = {}
        for model_type in models:
            y_encoded = None
            if labels is not None:
                le = LabelEncoder()
                y_encoded = le.fit_transform(labels)

            predictions = run_model(features_scaled, model_type, contamination, y_encoded, threshold_mode)
            df['anomaly_score'] = predictions
            anomalies = df[df['anomaly_score'] == 1]

            metrics = {'accuracy': None, 'precision': None, 'recall': None, 'f1': None}
            if labels is not None:
                try:
                    le = LabelEncoder()
                    y_true = le.fit_transform(labels)
                    y_pred = predictions.astype(int)
                    if len(le.classes_) == 2:
                        metrics = {
                            'accuracy': accuracy_score(y_true, y_pred),
                            'precision': precision_score(y_true, y_pred),
                            'recall': recall_score(y_true, y_pred),
                            'f1': f1_score(y_true, y_pred)
                        }
                except Exception as e:
                    print(f"Metrics error: {str(e)}")

            results[model_type] = {
                'metrics': metrics,
                'stats': {
                    'total_samples': len(df),
                    'normal': int(len(df) - len(anomalies)),
                    'anomalies': int(len(anomalies)),
                    'anomaly_rate': float(len(anomalies)) / len(df),
                    'used_contamination': contamination if model_type in ['isolation_forest', 'autoencoder'] and threshold_mode == 'contamination' else None
                }
            }

        return {
            'file_name': file.filename,
            'results': results,
            'preprocessed_download': f"/static/temp/{preprocessed_filename}"
        }

    except Exception as e:
        return {'error': str(e)}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files uploaded'}), 400
        files = request.files.getlist('files')
        if not files or all(file.filename == '' for file in files):
            return jsonify({'error': 'No files selected'}), 400

        models_str = request.form.get('models', '')
        target_col = request.form.get('targetColumn', '').strip()
        contamination_raw = request.form.get('contamination', '').strip()
        threshold_mode = request.form.get('thresholdMode', 'contamination')
        contamination = None

        try:
            if contamination_raw:
                contamination = float(contamination_raw)
                if not (0 < contamination < 1):
                    raise ValueError("Contamination must be between 0 and 1")
        except ValueError:
            return jsonify({'error': 'Invalid contamination value. Must be a float between 0 and 1.'}), 400

        if not models_str:
            return jsonify({'error': 'No models selected'}), 400

        models = models_str.split(',')
        valid_models = {'isolation_forest', 'autoencoder', 'random_forest', 'lightgbm', 'deep_learning'}
        invalid_models = set(models) - valid_models
        if invalid_models:
            return jsonify({'error': f'Invalid models: {", ".join(invalid_models)}'}), 400

        results, errors = [], []
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_file, file, models, target_col, contamination, threshold_mode) for file in files]
            for future in futures:
                try:
                    result = future.result()
                    if 'error' in result:
                        errors.append(result['error'])
                    else:
                        results.append(result)
                except Exception as e:
                    errors.append(str(e))
                    traceback.print_exc()

        if errors:
            return jsonify({
                'status': 'error',
                'error': 'Some files failed processing',
                'errors': errors,
                'results': results
            }), 400

        return jsonify(results)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

def cleanup_temp_folder():
    temp_dir = os.path.join(app.static_folder, 'temp')
    if os.path.exists(temp_dir):
        print(f"Cleaning up: {temp_dir}")
        shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)

if __name__ == '__main__':
    if sys.platform != 'win32':
        signal.signal(signal.SIGINT, lambda s, f: (cleanup_temp_folder(), sys.exit(0)))
        signal.signal(signal.SIGTERM, lambda s, f: (cleanup_temp_folder(), sys.exit(0)))
    else:
        import atexit
        atexit.register(cleanup_temp_folder)

    app.run(host='0.0.0.0', port=5000, debug=True, use_debugger=True, use_reloader=False if sys.platform == 'win32' else True)
