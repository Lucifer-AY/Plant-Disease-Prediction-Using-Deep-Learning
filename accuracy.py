import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model
model = tf.keras.models.load_model('trained_plant_disease_model.keras')

# Inspect the model
model.summary()

# Ensure test data is processed correctly
test_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)
test_data = test_data_gen.flow_from_directory(
    'test_data',
    target_size=(224, 224),  # Adjust as per your model's expected input size
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)



# Process predictions
predicted_classes = np.argmax(predictions, axis=1)  # Get predicted labels
true_labels = test_data.classes  # Ground truth labels

# Calculate accuracy
accuracy = accuracy_score(true_labels, predicted_classes)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix and Classification Report
cm = confusion_matrix(true_labels, predicted_classes)
class_labels = list(test_data.class_indices.keys())

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print("\nClassification Report:\n")
print(classification_report(true_labels, predicted_classes, target_names=class_labels))
