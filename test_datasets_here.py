import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("ecg_cnn_model.h5")
print("Model loaded successfully!")

# Load new preprocessed dataset
X_test = np.load("preprocessed_datasets/X.npy")  # Use the newly processed dataset
y_test = np.load("preprocessed_datasets/y.npy")  # Load the corresponding labels

# Evaluate model on the new dataset
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy on New Dataset: {test_acc:.4f}")

# ---- Make Predictions ----

# Predict probabilities
predictions = model.predict(X_test)
predicted_classes = (predictions > 0.5).astype(int)  # Convert probabilities to binary labels (0 or 1)

# Map index to class labels
class_mapping = {0: "Normal", 1: "Abnormal"}
predicted_labels = [class_mapping[cls[0]] for cls in predicted_classes]  # Convert to readable format

# Print results
for i, label in enumerate(predicted_labels[:10]):  # Display first 10 predictions
    print(f"Sample {i + 1}: Predicted Condition -> {label}")
