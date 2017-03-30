import  tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data # gets sample data from tensorflow website

mnist = input_data.read_data_sets("./data/", one_hot=True)

# number neurons per hidden layer
nodesLayer1 = 500
nodesLayer2 = 500
nodesLayer3 = 500

outputClasses = 10  #numbers 0-9

batchSize = 100

#Height x Width 28x28=784
x = tf.placeholder('float',[None,784]) #locks the data shape to exect data in this format
y = tf.placeholder('float')

def neuralNetworkModel(data):
    #(inputData * weights) + biases
    hidden1Layer ={'weights':tf.Variable(tf.random_normal([784, nodesLayer1])),
                   'biases':tf.Variable(tf.random_normal([nodesLayer1]))}

    hidden2Layer = {'weights': tf.Variable(tf.random_normal([nodesLayer1, nodesLayer2])),
                    'biases': tf.Variable(tf.random_normal([nodesLayer2]))}

    hidden3Layer = {'weights': tf.Variable(tf.random_normal([nodesLayer2, nodesLayer3])),
                    'biases': tf.Variable(tf.random_normal([nodesLayer3]))}

    outputLayer = {'weights': tf.Variable(tf.random_normal([nodesLayer3, outputClasses])),
                    'biases': tf.Variable(tf.random_normal([outputClasses]))}

    l1 = tf.add(tf.matmul(data,nodesLayer1['weights']) , hidden1Layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, nodesLayer2['weights']) , hidden2Layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, nodesLayer3['weights']) , hidden3Layer['biases'])
    l3 = tf.nn.relu(l3)

    outputLayer = tf.matmul(l3, outputLayer['weights']) + outputLayer['biases']

    return outputLayer


