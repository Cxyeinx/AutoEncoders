import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.nn import relu, softmax
import numpy as np
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = mnist.load_data()


x_train = x_train / 255.0
x_test = x_test / 255.0


model = Sequential()
model.add(Flatten(input_shape=(28,28,1)))
model.add(Dense(32, activation=relu))
model.add(Dense(32, activation=relu))
model.add(Dense(784, activation=relu))
model.add(Reshape((28,28,1)))


model.compile(optimizer="adam",
             metrics=["accuracy"],
             loss="binary_crossentropy")


model.fit(x_train, x_train, batch_size=32, epochs=10)


plt.imshow(x_test[0], cmap="gray")


x = x_test[0]
x = np.expand_dims(x, axis=-1)
x = np.expand_dims(x, axis=0)
y = model.predict(x)
y = np.squeeze(y, axis=0)
plt.imshow(y, cmap="gray")



