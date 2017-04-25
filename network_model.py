import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("./mnist_data/", one_hot = True)

with tf.name_scope("input"):
    x = tf.placeholder(shape = [None, 784], dtype = tf.float32, name = "x_input_slave")
    y = tf.placeholder(shape = [None, 10], dtype = tf.float32, name = "y_input_slave")


class Network:
    def __init__(self):
        self.layer1_nodes = 126
        self.layer2_nodes = 256
        self.layer3_nodes = 512
        self.batch_size = 16
        self.num_classes = 10
        self.num_epochs = 5
        self.accuracy = 0.0
        self.previous_accuracy = 0.0
        self.i = 1

    def set_learning_rate(self, learning_rate):
        self.learning_rate = tf.Variable(learning_rate, name = "Learning_rate_Slave")

    def get_weights(self, shape):
        return tf.Variable(tf.random_normal(shape))

    def get_biases(self, shape):
        return tf.Variable(tf.random_normal(shape))

    def conv2d(self, x, W):
        return tf.nn.conv2d(x ,W, strides = [1, 1, 1, 1], padding = "SAME")

    def max_pool(self, x):
        return tf.nn.max_pool(x , ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")


#        W_layer1 = self.get_weights([784, self.layer1_nodes])
#        b_layer1 = self.get_biases([self.layer1_nodes])

#        W_layer2 = self.get_weights([self.layer1_nodes, self.layer2_nodes])
#        b_layer2 = self.get_biases([self.layer2_nodes])

#        W_layer3 = self.get_weights([self.layer2_nodes, self.layer3_nodes])
#        b_layer3 = self.get_biases([self.layer3_nodes])

#        W_output_layer = self.get_weights([self.layer3_nodes, self.num_classes])
#        b_output_layer = self.get_biases([self.num_classes])


#        layer1 = tf.matmul(data, W_layer1) + b_layer1
#        layer2 = tf.matmul(layer1, W_layer2) + b_layer2
#        layer3 = tf.matmul(layer2, W_layer3) + b_layer3
#        output = tf.matmul(layer3, W_output_layer) + b_output_layer

    def neural_network(self, data):

        with tf.name_scope("Weights"):
            W_conv1 = self.get_weights([5, 5, 1, 32])
            W_conv2 = self.get_weights([5, 5, 32, 64])
            W_fc = self.get_weights([7*7*64, 1024])
            W_out = self.get_weights([1024, 10])

        with tf.name_scope("Biases"):
            b_conv1 = self.get_biases([32])
            b_conv2 = self.get_biases([64])
            b_fc = self.get_biases([1024])
            b_out = self.get_biases([10])

        data = tf.reshape(data, shape = [-1, 28, 28, 1])

        with tf.name_scope("Convolution_Layer"):
            conv1 = tf.nn.relu(self.conv2d(data, W_conv1) + b_conv1)
            conv1 = self.max_pool(conv1)
#            tf.summary.scalar("Convolution_Layer_1", conv1)

            conv2 = tf.nn.relu(self.conv2d(conv1, W_conv2) + b_conv2)
            conv2 = self.max_pool(conv2)
#            tf.summary.scalar("Convolution_Layer_2", conv2)

            fully_connected = tf.reshape(conv2, [-1, 7*7*64])
            fully_connected = tf.nn.relu(tf.matmul(fully_connected, W_fc) + b_fc)
            fully_connected = tf.nn.dropout(fully_connected, 0.8)
#            tf.summary.scalar("Fully_Connected_Layer", fully_connected)


            output = tf.matmul(fully_connected, W_out) + b_out

        return output

    def train_neural_network(self, x = x, save_model = True):
        prediction = self.neural_network(x)

        with tf.name_scope("Cost_Slave"):
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = y))
            tf.summary.scalar("Cost", cost)
        optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(cost)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            merged = tf.summary.merge_all()
            summary_file = "./summary/summary-{}".format(self.i)
            summary_writer = tf.summary.FileWriter(summary_file, sess.graph)
            accuracy_writer = tf.summary.FileWriter("./accuracy_summary/", sess.graph)

            print "\n***\nTraining Slave Network\n***\n"
            print "Episode : ",self.i
            print "Learning rate :", sess.run(self.learning_rate),"\n"
#            tf.summary.scalar("Accuracy", )
            for epoch in range(self.num_epochs):
                epoch_loss = 0

                for j in range(100):
                    epoch_x, epoch_y = mnist.train.next_batch(self.batch_size)

                    if j % 10 == 0:
                        summary ,_ , c = sess.run([merged, optimizer, cost], feed_dict = {x : epoch_x, y : epoch_y})
                        summary_writer.add_summary(summary, epoch*self.batch_size+j)
                    else:
                        _, c = sess.run([optimizer, cost], feed_dict = {x : epoch_x, y : epoch_y})

                    epoch_loss += c

                print "Epoch : ", epoch+1, " / ", self.num_epochs, ", Loss : ", epoch_loss

            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, dtype = tf.float32))
            self.accuracy = accuracy.eval({x:mnist.test.images, y:mnist.test.labels})
            tf.summary.scalar("Accuracy", self.accuracy)
            print "Accuracy : ", self.accuracy

            saver = tf.train.Saver()
            if save_model:
                if self.accuracy > self.previous_accuracy:
<<<<<<< HEAD
                    file_name = "./model/model-accuracy.ckpt"
=======
                    file_name = "./model/model-accuracy-{}.ckpt".format(self.accuracy)
>>>>>>> 7c19a9db0e6849d78a4888c7c020f8dfa317bc22
                    file_name = saver.save(sess, file_name)
                    print "Model saved in file ", file_name

        self.previous_accuracy = self.accuracy
        self.i += 1
        return self.accuracy


if __name__ == "__main__":
    network1 = Network()
    network1.set_learning_rate(1e-3)
    network1.train_neural_network(x)

    network1.set_learning_rate(1e-1)
    network1.train_neural_network(x)
