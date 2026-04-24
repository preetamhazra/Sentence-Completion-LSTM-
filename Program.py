# 1

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import nltk
import re

# 2

from google.colab import drive
drive.mount('/content/drive')

# update this path to your text file
file_path = '/content/drive/MyDrive/Shanku/Sentences.txt'

try:
    with open(file_path, 'r') as f:
        content = f.read()
        print("File opened successfully. First 100 characters:")
        print(content[:100])
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")

# 3

def remove_special_characters(text):
  # Removes emojis and special characters, keeping only alphanumeric and basic punctuation
  text = re.sub(r'[^\x00-\x7F]+', '', text)
  text = re.sub(r'[^a-zA-Z0-9\s.]', '', text)
  return text

def preprocess_text(text, low=True, sen=True):
  if low:
    text = text.lower()
  text = remove_special_characters(text)
  if sen:
    # Split by newline or period
    sentences = re.split(r'[\n.]', text)
    sentences = [s.strip() for s in sentences if s.strip()]
  return sentences


def remove_extra_spaces(text):
  text = re.sub(r'\s+', ' ', text)
  return text

def remove_extra_newlines(text):
  text = re.sub(r'\n+', '\n', text)
  return text

# 4

test_sample = """Hello World! 🌍 This is a test string @ 2024. It contains emojis 🚀 and some #special characters like %^&* symbols. Can the function handle this? Let's see... 12345. It should also split into sentences properly! 🔥"""

# Run the preprocessing function
cleaned_sentences = preprocess_text(content, low=True, sen=True)
#display(cleaned_sentences)

for i in range(len(cleaned_sentences)):
  cleaned_sentences[i] = remove_extra_spaces(cleaned_sentences[i])
  cleaned_sentences[i] = remove_extra_newlines(cleaned_sentences[i])


# 5

display(cleaned_sentences[0:2])

# 6

#display(cleaned_sentences)

# 7

# Tokenization
tokenizer=Tokenizer(oov_token='<oov>')
tokenizer.fit_on_texts(cleaned_sentences)

# 8

#tokenizer.word_counts

# 8

# Flatten all sentences into one continuous sequence
all_tokens = []

for line in cleaned_sentences:
    token_list = tokenizer.texts_to_sequences([line])[0]
    all_tokens.extend(token_list)

# Define sequence length
seq_len = 20  # try 20–30

input_sequences = []

for i in range(len(all_tokens) - seq_len):
    seq = all_tokens[i:i+seq_len+1]  # +1 for target
    input_sequences.append(seq)

print("Total sequences:", len(input_sequences))

# 9

# Padding
max_len=max([len(x) for x in input_sequences])


# 10

input_sequences = np.array(input_sequences)
#print(input_sequences)

# 11

# create input and output sequences
xs = input_sequences[:, :-1]
ys = input_sequences[:, 1:]

# 12

print(ys)

# 13

# split dataset into train, validation, and test sets.
from sklearn.model_selection import train_test_split
xs_train, xs_test, ys_train, ys_test = train_test_split(xs, ys, test_size=0.30, random_state=42)
xs_val, xs_test, ys_val, ys_test = train_test_split(xs_test, ys_test, test_size=0.50, random_state=42)

# 14

#print(ys_train)

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout, TimeDistributed # Import TimeDistributed
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping # Import EarlyStopping

# LSTM Model
total_words = len(tokenizer.word_index) + 1
model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_len-1))
model.add(Dropout(0.2)) # Added Dropout layer

# Layer 1
model.add(Bidirectional(LSTM(32, return_sequences=True))) # Added return_sequences=True
model.add(Dropout(0.2)) # Adde another Dropout layer

# Layer 2
model.add(Bidirectional(LSTM(32, return_sequences=True)))
model.add(Dropout(0.2)) # Added another Dropout layer

'''# Layer 3
model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(Dropout(0.2)) # Added another Dropout layer
'''
# Apply Dense layers to each timestep of the sequence output by the last LSTM
model.add(TimeDistributed(Dense(32, activation="relu"))) # Wrapped in TimeDistributed
model.add(TimeDistributed(Dense(total_words, activation='softmax'))) # Wrapped in TimeDistributed
adam=Adam(learning_rate=0.01)

# Define EarlyStopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    patience=20,          # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True # Restore model weights from the epoch with the best value of the monitored quantity
)

model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
history = model.fit(xs_train, ys_train, epochs=50, verbose=1, validation_data=(xs_val, ys_val), callbacks=[early_stopping]) # Add callbacks here




# Plot accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Plot loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()


# Perplexity
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Calculate perplexity
train_perplexity = np.exp(train_loss)
val_perplexity = np.exp(val_loss)

# Plot perplexity
plt.figure(figsize=(10, 6))
plt.plot(train_perplexity, label='Training Perplexity')
plt.plot(val_perplexity, label='Validation Perplexity')
plt.title('Model Perplexity')
plt.ylabel('Perplexity')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)
plt.show()
