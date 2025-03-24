import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# Check if a personalized model exists, otherwise load the base model
model_path = "ecg_cnn_model_personalized.h5" if os.path.exists("ecg_cnn_model_personalized.h5") else "ecg_cnn_model.h5"

try:
    model = load_model(model_path)
    print(f"Loaded model from {model_path} for further fine-tuning.")
    
    # **Recompile the model to reset the optimizer**
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    
except Exception as e:
    print("No existing model found or error in loading:", e)
    model = create_new_model()  # Define a new model from scratch

# Load newly recorded real-time ECG data
X_personal = np.load("preprocessed_datasets/X_personal.npy")
y_personal = np.load("preprocessed_datasets/y_personal.npy")

# Ensure NumPy format
X_personal = np.array(X_personal)
y_personal = np.array(y_personal)

# Reshape if needed (ensure it matches model input shape)
X_personal = X_personal.reshape(-1, X_personal.shape[1], 1)

# Train the model further on this personal data
model.fit(X_personal, y_personal, epochs=5, batch_size=8, verbose=1)

# Save the updated model
model.save("ecg_cnn_model_personalized.h5")
print("Model updated with personal ECG data and saved successfully!")
