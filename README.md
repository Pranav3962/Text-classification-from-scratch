
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>

<h1>Text Classification from Scratch</h1>

<h2>Project Overview</h2>
<p>This project demonstrates text sentiment classification starting from raw text files, specifically using the IMDB movie review dataset. The implementation leverages the <code>TextVectorization</code> layer for word splitting and indexing, and uses a simple 1D convolutional neural network (CNN) for classification. The project is optimized for GPU acceleration.</p>

<h2>Table of Contents</h2>
<ul>
    <li><a href="#introduction">Introduction</a></li>
    <li><a href="#setup">Setup</a></li>
    <li><a href="#data-preparation">Data Preparation</a></li>
    <li><a href="#data-loading">Data Loading</a></li>
    <li><a href="#data-preprocessing">Data Preprocessing</a></li>
    <li><a href="#model-building">Model Building</a></li>
    <li><a href="#training">Training</a></li>
    <li><a href="#evaluation">Evaluation</a></li>
    <li><a href="#end-to-end-model">End-to-End Model</a></li>
</ul>

<h2 id="introduction">Introduction</h2>
<p>This example shows how to perform text classification starting from raw text files on disk. We demonstrate the workflow using the unprocessed IMDB sentiment classification dataset and employ the <code>TextVectorization</code> layer for text preprocessing.</p>

<h2 id="setup">Setup</h2>
<p>To ensure the proper backend is used, set the Keras backend to TensorFlow:</p>
<pre><code>import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import tensorflow as tf
import numpy as np
from keras import layers
</code></pre>

<h2 id="data-preparation">Data Preparation</h2>
<p>First, download and extract the IMDB dataset:</p>
<pre><code>curl -O https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -xf aclImdb_v1.tar.gz
</code></pre>

<h2 id="data-loading">Data Loading</h2>
<p>Inspect the dataset structure:</p>
<pre><code>ls aclImdb
ls aclImdb/test
ls aclImdb/train
</code></pre>
<p>Remove the unsupervised training data:</p>
<pre><code>rm -r aclImdb/train/unsup
</code></pre>

<h2 id="data-preprocessing">Data Preprocessing</h2>
<p>Generate labeled datasets using <code>text_dataset_from_directory</code>:</p>
<pre><code>batch_size = 32
raw_train_ds = keras.utils.text_dataset_from_directory(
    "aclImdb/train",
    batch_size=batch_size,
    validation_split=0.2,
    subset="training",
    seed=1337,
)
raw_val_ds = keras.utils.text_dataset_from_directory(
    "aclImdb/train",
    batch_size=batch_size,
    validation_split=0.2,
    subset="validation",
    seed=1337,
)
raw_test_ds = keras.utils.text_dataset_from_directory(
    "aclImdb/test", batch_size=batch_size
)
</code></pre>

<h2 id="data-preprocessing">Data Preprocessing</h2>
<p>Define a custom standardization function to clean the text:</p>
<pre><code>import string
import re

def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "&lt;br /&gt;", " ")
    return tf.strings.regex_replace(
        stripped_html, f"[{re.escape(string.punctuation)}]", ""
    )
</code></pre>
<p>Create a <code>TextVectorization</code> layer and adapt it to the training data:</p>
<pre><code>max_features = 20000
embedding_dim = 128
sequence_length = 500

vectorize_layer = keras.layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode="int",
    output_sequence_length=sequence_length,
)

text_ds = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(text_ds)
</code></pre>
<p>Vectorize the data:</p>
<pre><code>def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label

train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

train_ds = train_ds.cache().prefetch(buffer_size=10)
val_ds = val_ds.cache().prefetch(buffer_size=10)
test_ds = test_ds.cache().prefetch(buffer_size=10)
</code></pre>

<h2 id="model-building">Model Building</h2>
<p>Build a simple 1D convolutional neural network:</p>
<pre><code>inputs = keras.Input(shape=(None,), dtype="int64")
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
</code></pre>

<h2 id="training">Training</h2>
<p>Train the model:</p>
<pre><code>epochs = 15
model.fit(train_ds, validation_data=val_ds, epochs=epochs)
</code></pre>

<h2 id="evaluation">Evaluation</h2>
<p>Evaluate the model on the test set:</p>
<pre><code>model.evaluate(test_ds)
</code></pre>

<h2 id="end-to-end-model">End-to-End Model</h2>
<p>Create an end-to-end model for processing raw strings:</p>
<pre><code>inputs = keras.Input(shape=(1,), dtype="string")
indices = vectorize_layer(inputs)
outputs = model(indices)

end_to_end_model = keras.Model(inputs, outputs)
end_to_end_model.compile(
    loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
)

end_to_end_model.evaluate(raw_test_ds)
</code></pre>

</body>
</html>
