import numpy as np
from tensorflow.keras import layers, models, optimizers, losses, metrics
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Read the dataset from the file
dataset = np.load('dataset.npz')
X_train = dataset['X_train']
y_train = dataset['y_train']
X_test = dataset['X_test']
y_test = dataset['y_test']

# Create a Machine Learning model for categorization of Devanagari Handwritten Characters
model = models.Sequential()
# Normalize the input data to have a mean of 0 and standard deviation of 1 for faster learning
model.add(layers.BatchNormalization(input_shape=(1024,)))
# Create a neural network with 3 layers
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
