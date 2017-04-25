import tensorflow as tf
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.losses import mean_squared_error
import network_model

network = network_model.Network()

memory = []
num_epochs_master = 10

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
<<<<<<< HEAD

    print "\n***\nTraining Master Network\n***\n"
=======
>>>>>>> 7c19a9db0e6849d78a4888c7c020f8dfa317bc22
    for batch in range(len(memory)-batch_size, len(memory)):
         accuracy, learning_rate, future_accuracy = recall(batch)
         epoch_accuracy.append(accuracy)
         epoch_learning_rate.append(learning_rate)
<<<<<<< HEAD
	 

	 model.fit(np.array([[accuracy]]), np.array([[learning_rate]]), epochs = num_epochs_master, shuffle = False)
#    print "Epoch Accuracy : ", epoch_accuracy
#    print ""
#    print "\n***\nTraining Master Network\n***\n"
#    model.fit(epoch_accuracy, epoch_learning_rate,epochs = num_epochs_master, batch_size = batch_size, shuffle = False)

def sort():
    for i in range(len(memory)):
	for j in range(len(memory) -i -1):
		if memory[j][2] > memory[j+1][2]:
			memory[j], memory[j+1] = memory[j+1], memory[j]
    print "\nMemory Sorted\n"	
=======

    print "\n***\nTraining Master Network\n***\n"
    model.fit(epoch_accuracy, epoch_learning_rate,epochs = num_epochs_master, batch_size = batch_size, shuffle = False)
>>>>>>> 7c19a9db0e6849d78a4888c7c020f8dfa317bc22


def recall(i):
    return memory[i]

def remember(accuracy, learning_rate, future_accuracy):
    memory.append((accuracy, learning_rate, future_accuracy))
    print "\nMemory : Accuracy -- ", accuracy
    print "Memory : Learning rate -- ", learning_rate
    print "Memory : Future accuracy -- ", future_accuracy
    print "Memory Size : ",len(memory), "\n"
<<<<<<< HEAD
    if len(memory) >= 2 :
	sort()
=======
>>>>>>> 7c19a9db0e6849d78a4888c7c020f8dfa317bc22


inital_accuracy = 0.8
accuracy = inital_accuracy

train_master_with_dummy()

<<<<<<< HEAD

=======
>>>>>>> 7c19a9db0e6849d78a4888c7c020f8dfa317bc22
while(True):

    prediction_rate = model.predict(np.array([accuracy]))
    prediction_rate = prediction_rate[0][0]

<<<<<<< HEAD
    if (prediction_rate < 0) or (prediction_rate > 10) :
=======
    if (prediction_rate < 0) and (prediction_rate > 10) :
>>>>>>> 7c19a9db0e6849d78a4888c7c020f8dfa317bc22
        continue

    network.set_learning_rate(prediction_rate)
    future_accuracy = network.train_neural_network()

    if future_accuracy > accuracy:
        remember(accuracy, prediction_rate, future_accuracy)

<<<<<<< HEAD
    if len(memory) % 3 == 0 and len(memory) != 0:
=======
    if len(memory) % 5 == 0 and len(memory) != 0:
>>>>>>> 7c19a9db0e6849d78a4888c7c020f8dfa317bc22
        train_master_network()


    accuracy = future_accuracy
