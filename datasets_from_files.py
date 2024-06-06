import os
import cv2 as cv
import numpy as np

X_train = []
y_train = []
X_test = []
y_test = []

# Traverse DevanagariHandwrittenCharacterDataset Test and Tran folders to create a dataset for learning
current_digit = 0
for i in range(10):
    for root, dirs, files in os.walk(f'DevanagariHandwrittenCharacterDataset/Test/digit_{i}'):
        print('Reading images from:', root)
        for file in files:
            # Read the image,convert it to grayscale and flatten it
            img = cv.cvtColor(cv.imread(os.path.join(root, file)), cv.COLOR_RGB2GRAY).flatten()
            # Add the image pixels to the test dataset
            X_test.append(img)
            y_test.append(current_digit)

    for root, dirs, files in os.walk(f'DevanagariHandwrittenCharacterDataset/Train/digit_{i}'):
        print('Reading images from:', root)
        for file in files:
            # Read the image,convert it to grayscale and flatten it
            img = cv.cvtColor(cv.imread(os.path.join(root, file)), cv.COLOR_RGB2GRAY).flatten()
            # Add the image pixels to the test dataset
            X_train.append(img)
            y_train.append(current_digit)
    current_digit += 1

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

# Get the indices of the train dataset
indices = np.arange(X_train.shape[0])
# Shuffle the indices
np.random.shuffle(indices)
# Shuffle the dataset using the shuffled indices
X_train = X_train[indices]
y_train = y_train[indices]

# Get the indices of the test dataset
indices = np.arange(X_test.shape[0])
# Shuffle the indices
np.random.shuffle(indices)
# Shuffle the dataset using the shuffled indices
X_test = X_test[indices]
y_test = y_test[indices]

# Save the dataset to a file
np.savez('dataset.npz', X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)