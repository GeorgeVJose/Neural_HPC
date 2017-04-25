import tensorflow as tf
import numpy as np
import network_model

network = network_model.Network()

memory = []
num_epochs_master = 5


def master_network(data):
    W_l1 = tf.Variable(tf.random_normal([1, 128]))
    b_l1 = tf.Variable(tf.random_normal([128]))

    W_l2 = tf.Variable(tf.random_normal([128, 256]))
    b_l2 = tf.Variable(tf.random_normal([256]))

    W_out = tf.Variable(tf.random_normal([256, 1]))
    b_out = tf.Variable(tf.random_normal([1]))

    l1 = tf.matmul(data, W_l1) + b_l1

    l2 = tf.matmul(l1, W_l2) + b_l2

    out = tf.matmul(l2, W_out) + b_out

    #Predicts learning_rate
    return out

def train_master_network():

    x = tf.placeholder(shape = [None], dtype = tf.float32, name = "master_x")
    y = tf.placeholder(shape = [None], dtype = tf.float32, name = "master_y")
    prediction = tf.placeholder(shape = [None], dtype = tf.float32, name= "master_prediction")

    batch_size = 5

    accuracy, learning_rate, future_accuracy = recall(np.random.randint(len(memory)))
    prediction = master_network([[accuracy]])

    print "Prediction : ",prediction.eval()[0][0]
    print "Learning_rate : ", learning_rate.eval()[0][0]


    print "\n***\nTraining Master Network\n***\n"

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        learning_rate = learning_rate.eval()[0][0]
        prediction = prediction.eval()[0][0]

        cost = tf.losses.mean_squared_error(labels = learning_rate, predictions =  prediction)
        optimizer = tf.train.AdamOptimizer(1e-3).minimize(cost)

        for epoch in range(num_epochs_master):
            batch_size = np.min(batch_size, len(memory))
            batches = np.random.choice(len(memory), batch_size)
            epoch_loss = 0

            for batch in batches:
                accuracy, learning_rate, future_accuracy = recall(batch)
                prediction = master_network([[accuracy]])

                _, c = sess.run([optimizer, cost], feed_dict = {x : [[accuracy]], y : learning_rate.eval()[0][0], prediction : prediction})
                epoch_loss += c

            print "Epoch : ",epoch+1, ". Loss : ", epoch_loss

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
i=0

while(True):
    prediction_rate = master_network([[accuracy]])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
#        prediction_rate = prediction_rate.eval()[0][0]
        if prediction_rate.eval()[0][0] < 0:
            continue
        network.set_learning_rate(prediction_rate.eval()[0][0])
    future_accuracy = network.train_neural_network()

    if future_accuracy > accuracy:
        remember(accuracy, prediction_rate, future_accuracy)

    if len(memory)%2 == 0 and len(memory) != 0:
        train_master_network()


    accuracy = future_accuracy
    i += 1
