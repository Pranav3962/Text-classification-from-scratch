import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import string
import re

# Set Keras backend to TensorFlow
os.environ["KERAS_BACKEND"] = "tensorflow"

# Download and extract the dataset using TensorFlow utilities
dataset_url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
dataset = tf.keras.utils.get_file("aclImdb_v1.tar.gz", dataset_url, untar=True, cache_dir=".")

# Define dataset directory
dataset_dir = os.path.join(os.path.dirname(dataset), "aclImdb")

# Remove the 'unsup' directory as it's not needed
unsup_dir = os.path.join(dataset_dir, 'train', 'unsup')
if os.path.exists(unsup_dir):
    tf.io.gfile.rmtree(unsup_dir)

# Set batch size
batch_size = 32

# Load dataset
raw_train_ds = keras.utils.text_dataset_from_directory(
    os.path.join(dataset_dir, 'train'),
    batch_size=batch_size,
    validation_split=0.2,
    subset="training",
    seed=1337,
)

raw_val_ds = keras.utils.text_dataset_from_directory(
    os.path.join(dataset_dir, 'train'),
    batch_size=batch_size,
    validation_split=0.2,
    subset="validation",
    seed=1337,
)

raw_test_ds = keras.utils.text_dataset_from_directory(
    os.path.join(dataset_dir, 'test'), batch_size=batch_size
)

# Print the number of batches in each dataset
print(f"Number of batches in raw_train_ds: {raw_train_ds.cardinality()}")
print(f"Number of batches in raw_val_ds: {raw_val_ds.cardinality()}")
print(f"Number of batches in raw_test_ds: {raw_test_ds.cardinality()}")

# Display a few samples from the training dataset
for text_batch, label_batch in raw_train_ds.take(1):
    for i in range(5):
        print(text_batch.numpy()[i])
        print(label_batch.numpy()[i])

# Custom standardization function
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    return tf.strings.regex_replace(
        stripped_html, f"[{re.escape(string.punctuation)}]", ""
    )

# Model constants
max_features = 20000
embedding_dim = 128
sequence_length = 500

# Text vectorization layer
vectorize_layer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode="int",
    output_sequence_length=sequence_length,
)

# Create a text-only dataset for vectorization
text_ds = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(text_ds)

# Function to vectorize text
def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label

# Vectorize the datasets
train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

# Prefetch the data for performance
train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# Build the model
inputs = keras.Input(shape=(None,), dtype="int64")
x = layers.Embedding(max_features, embedding_dim)(inputs)
x = layers.Dropout(0.5)(x)
x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.5)(x)
predictions = layers.Dense(1, activation="sigmoid", name="predictions")(x)

model = keras.Model(inputs, predictions)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
epochs = 15
model.fit(train_ds, validation_data=val_ds, epochs=epochs)

# Evaluate the model
model.evaluate(test_ds)

# Build an end-to-end model for prediction
inputs = keras.Input(shape=(1,), dtype="string")
indices = vectorize_layer(inputs)
outputs = model(indices)
end_to_end_model = keras.Model(inputs, outputs)
end_to_end_model.compile(
    loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
)

# Evaluate the end-to-end model
end_to_end_model.evaluate(raw_test_ds)

# Example text to classify
texts = [
    "I really loved this movie! It was fantastic and the performances were great.",
    "movie was very bad"
]

# Convert the list of texts into a numpy array with shape (number_of_texts, 1)
text_array = np.array(texts, dtype=object)[:, np.newaxis]

# Use the end-to-end model to make predictions
predictions = end_to_end_model.predict(text_array)

# Print the results
for i, text in enumerate(texts):
    print(f"Text: {text}")
    print(f"Prediction (0 = negative, 1 = positive): {predictions[i][0]:.4f}\n")
