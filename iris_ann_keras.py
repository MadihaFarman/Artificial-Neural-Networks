"""
ANN on the Iris dataset with TensorFlow / Keras

Task: classify an iris flower into one of 3 species from 4 measurements.
Type of problem: multi-class classification (3 classes, exactly one is right).

Install once if needed:
    pip install tensorflow scikit-learn pandas
"""

import joblib
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# make results repeatable (same weights at the start, same split every run)
SEED = 42
keras.utils.set_random_seed(SEED)

print("TensorFlow:", tf.__version__)

# ---------------------------------------------------------------------------
# 1. Load the data and look at it
# ---------------------------------------------------------------------------
iris = load_iris(as_frame=True)

df = iris.frame.copy()                                   # features + 'target'
class_names = [str(n) for n in iris.target_names]        # setosa, versicolor, virginica

# readable species column (only for looking at the data)
df["species"] = df["target"].map(dict(enumerate(class_names)))

print("\nShape (rows, columns):", df.shape)
print("Classes:", class_names)
print("\nFirst 5 rows:")
print(df.head())
print("\nRandom sample from all species:")
print(df.sample(8, random_state=SEED))
print("\nLast 5 rows:")
print(df.tail())

# ---------------------------------------------------------------------------
# 2. Explore the data
# ---------------------------------------------------------------------------
print("\nData info:")
df.info()

print("\nSummary statistics:")
print(df.drop(columns=["target"]).describe().T)

print("\nClass balance:")
print(df["species"].value_counts())

# ---------------------------------------------------------------------------
# 3. Clean the data
# ---------------------------------------------------------------------------
print("\nMissing values per column:")
print(df.isna().sum())
print("Total missing:", df.isna().sum().sum())

print("\nNumber of duplicated rows:", df.duplicated().sum())
print(df[df.duplicated(keep=False)])
# Decisions: no missing values, all numeric, duplicate pair kept (two different
# flowers can have identical measurements), outliers kept (real measurements),
# classes balanced (50/50/50) so no resampling is needed.

# ---------------------------------------------------------------------------
# 4. Prepare the data
# ---------------------------------------------------------------------------
# 4.1 Separate features and labels
X = df[iris.feature_names].values.astype("float32")      # shape (150, 4)
y = df["target"].values                                  # shape (150,), values 0, 1, 2

print("\nX shape:", X.shape)
print("y shape:", y.shape)
print("labels :", np.unique(y), "->", class_names)

# 4.2 Split into train / validation / test (60 / 20 / 20), stratified
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.40, stratify=y, random_state=SEED)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED)

print("\ntrain:", X_train.shape, " val:", X_val.shape, " test:", X_test.shape)
print("class counts in train:", np.bincount(y_train))
print("class counts in val  :", np.bincount(y_val))
print("class counts in test :", np.bincount(y_test))

# 4.3 Scale the features (fit on TRAIN only to avoid data leakage)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

print("\nMean per feature in train after scaling:", X_train_s.mean(axis=0).round(3) + 0.0)
print("Std  per feature in train after scaling:", X_train_s.std(axis=0).round(3))

# 4.4 Labels stay as integers (0, 1, 2) -> use sparse_categorical_crossentropy

# ---------------------------------------------------------------------------
# 5. Build the ANN
#    4 inputs -> Dense(16, relu) -> Dense(8, relu) -> Dense(3, softmax)
# ---------------------------------------------------------------------------
model = keras.Sequential([
    keras.Input(shape=(4,)),                  # 4 features per flower
    layers.Dense(16, activation="relu"),      # hidden layer 1
    layers.Dense(8, activation="relu"),       # hidden layer 2
    layers.Dense(3, activation="softmax"),    # output: probability of each species
])

model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# ---------------------------------------------------------------------------
# 6. Train
# ---------------------------------------------------------------------------
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=20,
    restore_best_weights=True,
)

history = model.fit(
    X_train_s, y_train,
    validation_data=(X_val_s, y_val),
    epochs=300,              # upper limit; early stopping usually stops sooner
    batch_size=16,
    callbacks=[early_stop],
    verbose=0,
)

print("\nStopped after", len(history.history["loss"]), "epochs")
print("Best validation loss:", round(min(history.history["val_loss"]), 4))

# ---------------------------------------------------------------------------
# 7. Evaluate on the test set (looked at once, at the end)
# ---------------------------------------------------------------------------
test_loss, test_acc = model.evaluate(X_test_s, y_test, verbose=0)
print(f"\nTest loss    : {test_loss:.4f}")
print(f"Test accuracy: {test_acc:.4f}")

probs = model.predict(X_test_s, verbose=0)    # shape (30, 3), each row sums to 1
y_pred = np.argmax(probs, axis=1)             # class with the highest probability

print("\nFirst 5 probability rows (setosa, versicolor, virginica):")
print(probs[:5].round(3))
print("Predicted classes:", y_pred[:5])
print("True classes     :", y_test[:5])

print("\nConfusion matrix (rows = true, columns = predicted):")
print(pd.DataFrame(confusion_matrix(y_test, y_pred),
                   index=class_names, columns=class_names))

print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=class_names, digits=3))

# ---------------------------------------------------------------------------
# 8. Predict on a new flower
# ---------------------------------------------------------------------------
# sepal length, sepal width, petal length, petal width (in cm)
new_flower = np.array([[5.9, 3.0, 5.1, 1.8]], dtype="float32")

new_scaled = scaler.transform(new_flower)     # use the SAME scaler from training
p = model.predict(new_scaled, verbose=0)[0]

print("New flower probabilities:")
for name, prob in zip(class_names, p):
    print(f"  {name:<11} {prob:.3f}")
print("Predicted species:", class_names[int(np.argmax(p))])

# Save the model and the scaler (both are needed to reuse the model later)
model.save("iris_ann.keras")
joblib.dump(scaler, "iris_scaler.joblib")
print("\nsaved: iris_ann.keras, iris_scaler.joblib")
