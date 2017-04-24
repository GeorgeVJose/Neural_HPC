import tensorflow as tf
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.losses import mean_squared_error
import network_model

network = network_model.Network()

memory = []
num_epochs_master = 5

x = tf.placeholder(shape = [None, 1], dtype = tf.float32, name = "master_x")
y = tf.placeholder(shape = [None, 1], dtype = tf.float32, name = "master_y")
prediction = tf.placeholder(shape = [None, 1], dtype = tf.float32, name= "master_prediction")


model = Sequential()

model.add(Dense(units = 128, input_dim = 1))
model.add(Dense(units = 256))
model.add(Dense(units = 1))

model.compile(loss = 'mean_squared_error', optimizer = 'Adam', metrices = ['accuracy'])


def get_dummy_data(size = 100):
    y = np.random.random_sample(size)/10
    x =[]
    while(len(x) != size):
        temp = np.random.random_sample()
        if temp > 0.8:
            x.append(temp)
    return x, y


def train_master_with_dummy():
     x_data, y_data = get_dummy_data()

     print "\n***\nTraining Master network with dummy values\n***\n"
     model.fit(x_data, y_data, epochs = 10, batch_size = 10)
     print "Done!."

#     print "Prediction : ", model.predict(np.array([0.85]), batch_size = 1)


def train_master_network():
    batch_size = 5
    batch_size = np.min([batch_size, len(memory)])

    epoch_accuracy = []
    epoch_learning_rate = []
    for batch in range(len(memory)-batch_size, len(memory)):
         accuracy, learning_rate, future_accuracy = recall(batch)
         epoch_accuracy.append(accuracy)
         epoch_learning_rate.append(learning_rate)

    print "\n***\nTraining Master Network\n***\n"
    model.fit(epoch_accuracy, epoch_learning_rate, batch_size = batch_size, shuffle = False)


def recall(i):
    return memory[i]

def remember(accuracy, learning_rate, future_accuracy):
    memory.append((accuracy, learning_rate, future_accuracy))
    print "\nMemory : Accuracy -- ", accuracy
    print "Memory : Learning rate -- ", learning_rate
    print "Memory : Future accuracy -- ", future_accuracy
    print "Memory Size : ",len(memory), "\n"


inital_accuracy = 0.8
accuracy = inital_accuracy

train_master_with_dummy()

while(True):

    prediction_rate = model.predict(np.array([accuracy]))
    prediction_rate = prediction_rate[0][0]

    if (prediction_rate < 0) and (prediction_rate > 10) :
        continue

    network.set_learning_rate(prediction_rate)
    future_accuracy = network.train_neural_network()

    if future_accuracy > accuracy:
        remember(accuracy, prediction_rate, future_accuracy)

    if len(memory) % 5 == 0 and len(memory) != 0:
        train_master_network()


    accuracy = future_accuracy
