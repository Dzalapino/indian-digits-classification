import os
import cv2 as cv
import numpy as np
from tensorflow.keras import layers, models, optimizers, losses, metrics
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt

# X_train = []
# y_train = []
# X_test = []
# y_test = []
#
# # Traverse DevanagariHandwrittenCharacterDataset Test and Tran folders to create a dataset for learning
# current_digit = 0
# for i in range(10):
#     for root, dirs, files in os.walk(f'DevanagariHandwrittenCharacterDataset/Test/digit_{i}'):
#         print('Reading images from:', root)
#         for file in files:
#             # Read the image,convert it to grayscale and flatten it
#             img = cv.cvtColor(cv.imread(os.path.join(root, file)), cv.COLOR_RGB2GRAY).flatten()
#             # Add the image pixels to the test dataset
#             X_test.append(img)
#             y_test.append(current_digit)
#
#     for root, dirs, files in os.walk(f'DevanagariHandwrittenCharacterDataset/Train/digit_{i}'):
#         print('Reading images from:', root)
#         for file in files:
#             # Read the image,convert it to grayscale and flatten it
#             img = cv.cvtColor(cv.imread(os.path.join(root, file)), cv.COLOR_RGB2GRAY).flatten()
#             # Add the image pixels to the test dataset
#             X_train.append(img)
#             y_train.append(current_digit)
#     current_digit += 1
#
# X_train = np.array(X_train)
# y_train = np.array(y_train)
# X_test = np.array(X_test)
# y_test = np.array(y_test)
#
# # Get the indices of the train dataset
# indices = np.arange(X_train.shape[0])
# # Shuffle the indices
# np.random.shuffle(indices)
# # Shuffle the dataset using the shuffled indices
# X_train = X_train[indices]
# y_train = y_train[indices]
#
# # Get the indices of the test dataset
# indices = np.arange(X_test.shape[0])
# # Shuffle the indices
# np.random.shuffle(indices)
# # Shuffle the dataset using the shuffled indices
# X_test = X_test[indices]
# y_test = y_test[indices]

# # Save the dataset to a file
# np.savez('dataset.npz', X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

# Read the dataset from the file
dataset = np.load('dataset.npz')
X_train = dataset['X_train']
y_train = dataset['y_train']
X_test = dataset['X_test']
y_test = dataset['y_test']

# Create a Machine Learning model for categorization of Devanagari Handwritten Characters
model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(1024,)))
model.add(layers.Dropout(0.15))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.15))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer=optimizers.Adam(), loss=losses.sparse_categorical_crossentropy,
              metrics=[metrics.sparse_categorical_accuracy])

model.fit(X_train, y_train, epochs=6, batch_size=128, validation_data=(X_test, y_test))

# Show loss and accuracy
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')

# Show some predictions with corresponding digit images and actual labels
predictions = model.predict(X_test)
for i in range(10):
    n = np.random.randint(0, X_test.shape[0])
    plt.imshow(X_test[n].reshape(32, 32), cmap='gray')
    plt.title(f'Predicted: {np.argmax(predictions[n])}, Actual: {y_test[n]}')
    plt.show()

# Show confusion matrix
confusion = confusion_matrix(y_test, np.argmax(predictions, axis=1))
plt.matshow(confusion)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# Show f1 score
f1 = f1_score(y_test, np.argmax(predictions, axis=1), average='micro')
print(f'F1 score: {f1}')

print('SVM APPROACH')
# Try with SVM
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

svm = make_pipeline(StandardScaler(), SVC())
svm.fit(X_train, y_train)
print(svm.score(X_test, y_test))

# Show SVM predictions with corresponding digit images and actual labels
predictions = svm.predict(X_test)
for i in range(10):
    n = np.random.randint(0, X_test.shape[0])
    plt.imshow(X_test[n].reshape(32, 32), cmap='gray')
    plt.title(f'Predicted: {predictions[n]}, Actual: {y_test[n]}')
    plt.show()

# Show SVM confusion matrix
confusion = confusion_matrix(y_test, predictions)
plt.matshow(confusion)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# Show SVM f1 score
f1 = f1_score(y_test, predictions, average='micro')
print(f'F1 score: {f1}')
