# Import necessary libraries
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the dataset of video game reviews
df = pd.read_csv('reviews.csv')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['sentiment'], test_size=0.2)

# Tokenize the text data
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Pad the sequences to the same length
max_length = max([len(x) for x in X_train_seq])
X_train_seq = pad_sequences(X_train_seq, maxlen=max_length)
X_test_seq = pad_sequences(X_test_seq, maxlen=max_length)

# Build the model
model = keras.Sequential([
    # Embed the text data into a lower-dimensional space
    keras.layers.Embedding(10000, 16),
    # Average the embeddings across all words in each review
    keras.layers.GlobalAveragePooling1D(),
    # Add a fully-connected layer with ReLU activation
    keras.layers.Dense(16, activation='relu'),
    # Add a final sigmoid output layer with three units (one for each class: positive, neutral, negative)
    keras.layers.Dense(3, activation='sigmoid')
])

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model on the training data
history = model.fit(X_train_seq, y_train, epochs=5, validation_data=(X_test_seq, y_test))

# Evaluate the model on the testing data
test_loss, test_acc = model.evaluate(X_test_seq, y_test)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)

new_text = 'This game was so much fun! I can't wait to play it again!'
new_seq = tokenizer.texts_to_sequences([new_text])
new_seq = pad_sequences(new_seq, maxlen=max_length)
prediction = model.predict(new_seq)

# Print the predicted class
if prediction[0][0] > prediction[0][1] and prediction[0][0] > prediction[0][2]:
    print('Positive')
elif prediction[0][1] > prediction[0][0] and prediction[0][1] > prediction[0][2]:
    print('Neutral')
else:
    print('Negative')
