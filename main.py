import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

global mnist
# load the dataset of handwritten digits (about 60k samples)
mnist = tf.keras.datasets.mnist
# split it into two tuples - one for training and one for testing
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# normalize (scale down) the training data to be between 0 and 1 so that it's easier to compute
# we don't scale the y data, because these are the labels (0, 1, 2, 3...)
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)
# create a basic neural network
model = tf.keras.models.Sequential()
# Create the input layer. Since the image is 28x28 pixels, it is flattened to 784 neurons
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
# Create two hidden layers that are exactly the same. Each has 128 neurons.
# "Dense" layer means that all the neurons are connected to the previous and the next layer
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
# Create the output layer. It has 10 neurons (for each digit) and scales down the values so they all add up to 1
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Train the model. 'epochs' denotes how many times are we going to repeat the training process
model.fit(x_train, y_train, epochs=3)
accuracy, loss = model.evaluate(x_test, y_test)
print(accuracy)
print(loss)
# Save the trained model, so that we can load it later and feed our own images into the neural network
model.save('digits.model')

for x in range(1, 8):
    img = cv.imread(f'{x}.png')[:, :, 0]
    # invert is needed to produce black on white
    img = np.invert(np.array([img]))
    prediction = model.predict(img)
    print(f'The digit on the pic is probably: {np.argmax(prediction)}')
    # binary cmap is in order to make the image black and white
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()
