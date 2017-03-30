import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data  # gets sample data from tensorflow website

mnist = input_data.read_data_sets("./data/", one_hot=True)

inputsPerRow = 784

layer1Neurons = 500
layer2Neurons = 500
layer3Neurons = 500

outputClasses = 10
miniBatchSize = 100

# Height x Width 28x28=784
dataRowInput = tf.placeholder('float', [None, inputsPerRow])  #data
outputLabel = tf.placeholder('float') #labels

def neural_network_model(dataRow):

    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([inputsPerRow, layer1Neurons])), 'biases': tf.Variable(tf.random_normal([layer1Neurons]))}

    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([layer1Neurons, layer2Neurons])), 'biases': tf.Variable(tf.random_normal([layer2Neurons]))}

    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([layer2Neurons, layer3Neurons])), 'biases': tf.Variable(tf.random_normal([layer3Neurons]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([layer3Neurons, outputClasses])), 'biases': tf.Variable(tf.random_normal([outputClasses]))}



    l1 = tf.matmul(dataRow, hidden_1_layer['weights']) + hidden_1_layer['biases']
    l1 = tf.nn.relu(l1)

    l2 = tf.matmul(l1, hidden_2_layer['weights']) + hidden_2_layer['biases']
    l2 = tf.nn.relu(l2)

    l3 = tf.matmul(l2, hidden_3_layer['weights']) + hidden_3_layer['biases']
    l3 = tf.nn.relu(l3)

    outputVector = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return outputVector


def trainNet(triningRow):
    probabilityPrediction = neural_network_model(triningRow)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=probabilityPrediction, labels=outputLabel))
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    totalEpocs = 10

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoc in range(totalEpocs):
            epoch_loss = 0
            for numberOfMiniBatches in range(int(mnist.train.num_examples / miniBatchSize)):
                trainingRows, trainingLables = mnist.train.next_batch(miniBatchSize)
                numberOfMiniBatches, miniBatchCost = sess.run([optimizer, loss], feed_dict={triningRow: trainingRows, outputLabel: trainingLables})
                epoch_loss += miniBatchCost
            print('Epoc', epoc, ' completed out of ', totalEpocs, 'Loss:', epoch_loss)

        correct = tf.equal(tf.argmax(probabilityPrediction, 1), tf.argmax(outputLabel, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        print('Accuracy: ', accuracy.eval({triningRow: mnist.test.images, outputLabel: mnist.test.labels}))


trainNet(dataRowInput)
