import tensorflow as tf
import numpy as np
import network_model

network = network_model.Network()

memory = []
num_epochs_master = 5

x = tf.placeholder(shape = [None, 1], dtype = tf.float32, name = "master_x")
y = tf.placeholder(shape = [None, 1], dtype = tf.float32, name = "master_y")
prediction = tf.placeholder(shape = [None, 1], dtype = tf.float32, name= "master_prediction")

def master_network(data):
    W_l1 = tf.Variable(tf.random_normal([1, 64]))
    b_l1 = tf.Variable(tf.random_normal([64]))

    W_l2 = tf.Variable(tf.random_normal([64, 128]))
    b_l2 = tf.Variable(tf.random_normal([128]))

    W_out = tf.Variable(tf.random_normal([128, 1]))
    b_out = tf.Variable(tf.random_normal([1]))

    l1 = tf.matmul(data, W_l1) + b_l1

    l2 = tf.matmul(l1, W_l2) + b_l2

    out = tf.matmul(l2, W_out) + b_out

    #Predicts learning_rate
    return out

def get_dummy_data():
    x = np.random.random_sample(size = 100)/10
    y =[]
    while(len(y) != 100):
        temp = np.random.random_sample()
        if temp > 0.8:
            y.append(temp)
    return x, y


def train_master_with_dummy():
     x_data, y_data = get_dummy_data()
     prediction = master_network(x)
     cost = tf.losses.mean_squared_error(labels = y, predictions = prediction)
     optimizer = tf.train.AdamOptimizer(1e-3).minimize(cost)

     with tf.Session().as_default() as sess:
         sess.run(tf.global_variables_initializer())

         print "\n***\nTraining Master network with dummy values\n***\n"

         for epoch in range(5):
             for _ in range(5000):
                 i = np.random.randint(0, 100)
                 epoch_x = x_data[i]
                 epoch_y = y_data[i]

                 _, c = sess.run([optimizer, cost], feed_dict = {x : [[epoch_x]], y : [[epoch_y]]})
             print "Epoch : ", epoch+1

     print "Done!."

def train_master_network():
    accuracy, learning_rate, future_accuracy = recall(np.random.randint(len(memory)))
#    prediction = master_network(x)
    prediction = tf.run()
#    print "Shape : ", np.shape([[accuracy]])
#    print "Shape : ", np.shape(learning_rate)

    batch_size = 5

    cost = tf.losses.mean_squared_error(labels = y, predictions =  prediction)
    optimizer = tf.train.AdamOptimizer(1e-3).minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        print "\n***\nTraining Master Network\n***\n"

        batch_size = np.min([batch_size, len(memory)])
#        batches = np.random.choice(len(memory), batch_size)
#        batches =
        for epoch in range(num_epochs_master):
            epoch_loss = 0

            for batch in range(len(memory)-batch_size, len(memory)):
                accuracy, learning_rate, future_accuracy = recall(batch)
                prediction = master_network([[accuracy]])

#                print "Master, Accuracy : ", [[accuracy]]
#                print "Master, Learning_rate : ", [[learning_rate]]
#                print "Master, Prediction"
                _, c = sess.run([optimizer, cost], feed_dict = {x : np.array([[accuracy]]), y : np.array([[learning_rate]])})
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

train_master_with_dummy()

i=1
while(i >= 1):
    prediction_rate = master_network([[accuracy]])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        prediction_rate = prediction_rate.eval()[0][0]

#    print i, " prediction_rate : ",prediction_rate
    if (prediction_rate < 0) and (prediction_rate > 10) :
        continue

    network.set_learning_rate(prediction_rate)
    future_accuracy = network.train_neural_network()

    if future_accuracy > accuracy:
        remember(accuracy, prediction_rate, future_accuracy)

    if len(memory) % 5 == 0 and len(memory) != 0:
        train_master_network()


    accuracy = future_accuracy
    i += 1
