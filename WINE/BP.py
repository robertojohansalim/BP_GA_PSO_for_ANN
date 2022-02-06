# %%
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# %%
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def load_datasets(datasets_name="wine"):
    if datasets_name == "iris":
        dataset = datasets.load_iris()
    elif datasets_name == "wine":
        dataset = datasets.load_wine()

    X = dataset.data
    y = dataset.target

    y_ = y.reshape(-1, 1) # Convert data to a single column

    # One Hot encode the class labels
    encoder = OneHotEncoder(sparse=False)
    encoder.fit(y_)
    y = encoder.transform(y_)
    print(encoder.categories_)
    print(len(encoder.categories_[0]))

    # Split the data for training and testing
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    return x_train, x_test, y_train, y_test

def saveReport(accuracy_history):
    data = {
        "accuracy_history": accuracy_history
    }
    pickle.dump(data, open("./BP_report.pkl", "wb"))

x_train, x_test, y_train, y_test = load_datasets()

# %%
# Build the model

model = Sequential()

model.add(Dense(20, input_shape=(x_train[0].shape[0],), activation='relu', name='fc1'))
model.add(Dense(3, activation='softmax', name='output'))

# Adam optimizer with learning rate of 0.001
optimizer = Adam(lr=0.01)
model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

print('Neural Network Model Summary: ')
print(model.summary())

# Train the model
print("X Train Shape",x_train.shape)
print("X Train Elemn",x_train[0])
print("Y Train Shape:",y_train.shape)
print("Y Train Elemn:",y_train[0])
print("Input Shape:", model.input.shape)
train_history = model.fit(x_train, y_train, verbose=2, batch_size=5, epochs=1000)

plt.plot(train_history.history["accuracy"])
plt.show(block=False)

# %%
# Test on unseen data

results = model.evaluate(x_test, y_test)

print("Evaluating Model Using test splits dataset")
print('Final test set loss: {:4f}'.format(results[0]))
print('Final test set accuracy: {:4f}'.format(results[1]))


print("Model input:", model.input.shape)
print('Example Prediction from test split: ')
# for idx, x in enumerate(zip(x_test[:5], y_test[:5])):
#     plt.subplot("15{}".format(idx))
#     pred = model.predict(np.array([x[0]]))
#     pred = le.inverse_transform(pred)
#     truth = le.inverse_transform([x[1]])
#     plt.title("{}:{}".format(truth[0], pred[0]))
#     plt.imshow(np.array(x[0]).reshape((28,28,1)))
plt.show(block=False)

saveReport(train_history.history["accuracy"])


