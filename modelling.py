import mlflow.sklearn
import pandas as pd
import mlflow
from sklearn.linear_model import SGDClassifier
from joblib import dump

# Memuat dataset
data = pd.read_csv("wine.csv")

# Model Online Learning
model = SGDClassifier(
    loss='log_loss',
    learning_rate='adaptive',
    eta0=0.01,
    max_iter=10000
)

# Kelas target
classes = data['target'].unique()

# Aktifkan autolog
mlflow.autolog()

# Preprocessing data
X_batch = data.drop(columns=['target'])
y_batch = data['target']

# Training
model.partial_fit(X_batch, y_batch, classes=classes)

# Evaluasi
accuracy = model.score(X_batch, y_batch)
mlflow.log_metric("accuracy", accuracy)

print(f"Accuracy: {accuracy:.4f}")

# Simpan model
dump(model, "online_model.joblib")

# Log artifact
mlflow.log_artifact(
    "online_model.joblib",
    artifact_path="model_artifacts"
)

# Log model
mlflow.sklearn.log_model(
    sk_model=model,
    artifact_path="online_model",
    input_example=X_batch.iloc[:5]
)