'''
Huge part of this is from: https://www.youtube.com/watch?v=BhpvH5DuVu8
i just added the option to change the number of hidden layers and number of nodes
i experimented with some optimizers; seems like some do fine under certain conditions

+
made the model actually usable and saving the epoch states
'''


import tempfile
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os
import random


def model(data):
    layers = []
    for i in range(0, n_layers):
        if i == 0:
            l = {
                'weights': tf.Variable(tf.random_normal([inputsize, n_nodes[i]])),
                'biases': tf.Variable(tf.random_normal([n_nodes[i]]))
            }
        else:
            l = {
                'weights': tf.Variable(tf.random_normal([n_nodes[i-1], n_nodes[i]])),
                'biases': tf.Variable(tf.random_normal([n_nodes[i]]))
            }
        layers.append(l)
    output_layer =  {
        'weights': tf.Variable(tf.random_normal([n_nodes[len(n_nodes) - 1], n_classes])),
        'biases': tf.Variable(tf.random_normal([n_classes]))
    }
    layers.append(output_layer)
    res = []
    for i in range(0, n_layers):
        a = layers[i]
        if i == 0:
            lx = tf.add(tf.matmul(data, a['weights']), a['biases'])
        else:
            lx = tf.add(tf.matmul(res[i-1], a['weights']), a['biases'])
        lx = tf.nn.relu(lx)
        res.append(lx)
    return tf.matmul(res[len(res) - 1], layers[len(layers) - 1]['weights']) + layers[len(layers) - 1]['biases']
   

def train(x):
    prediction = model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

    optimizer = tf.train.AdamOptimizer().minimize(cost)
    # Those are some other optimizer. found out that Adam does pretty good
    # although Adagrad seems to be fine too.
    #optimizer = tf.train.AdagradOptimizer(0.001).minimize(cost)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs_no):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict = {
                    x: epoch_x, 
                    y: epoch_y
                })
                epoch_loss += c
            print('Saving current epoch to model...')
            tf.train.Saver().save(sess, "./model.ckpt")
            print('Epoch =>', epoch + 1, '/', epochs_no, 'loss =>', epoch_loss)      
        correct = tf.equal(tf.argmax(prediction, axis=1), tf.argmax(y, axis=1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        testsize = int(mnist.test.images.shape[0] * test_set_size_percent)
        print("Testing Accuracy with", testsize, "Elements of test images")
        print('Accuracy => ', accuracy.eval({x: mnist.test.images[:testsize], y: mnist.test.labels[:testsize]}))

def use(data):
    prediction = model(x)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(epochs_no):
            try:
                tf.train.Saver().restore(sess, "./model.ckpt")
            except Exception as err:
                print(str(err))
            epoch_loss = 0

        result = (sess.run(tf.argmax(prediction.eval(feed_dict={
            x: data
        }), axis=1)))
        return result[0]




# width * height of each image
inputsize = 28 * 28 

# how much of the test images should be used for accuracy
test_set_size_percent = 0.5

#how many 'rounds' should we go?
epochs_no = 10

x = tf.placeholder('float', [None, inputsize])
y = tf.placeholder('float')


#load mnist
tmp = tempfile.gettempdir() + '/mnist_data'
print(tmp)
mnist = input_data.read_data_sets(tmp, one_hot=True)
'''
one hot means [1,0,0,0,0,0,0,0,0,0] -> its the digit 0
'''

#how many numbers do we got?
n_classes = 10

# how many items do we want per batch?
batch_size = 100

#how many nodes per layer?
n_nodes = [500,500,500]
#how many layers do we need?
n_layers = len(n_nodes)

# ONLY TRAIN ONCE TO GENERATE THE MODEL
if not os.path.isfile('./model.ckpt.index'):
    print("Pretrained model does not exist...training...")
    print("Please wait...KTHX")
    train(x)
    print("Launch the program again to check some Numbers ;)")
    print("KTHXBYE")
else:
    print("Pretrained model exists...now using the NN")
    #
    #
    # THIS IS A DEMONSTRATION ON HOW TO ACTUALLY USE THE NN
    #
    # test with a test image outside of the ones we used for accuracy
    start = int(mnist.test.images.shape[0] * test_set_size_percent)
    SAMPLE_NUM = random.randint(start,mnist.test.images.shape[0])
    print("Picking Sample with index " + str(SAMPLE_NUM))
    use_test_sample = np.matrix(mnist.test.images[SAMPLE_NUM])
    for i,l in enumerate(mnist.test.labels[SAMPLE_NUM]):
        if l == 1.0:
            use_test_label = i

    classification = use(use_test_sample)
    print("We got " + str(classification) + "<=>(Real):" + str(use_test_label))
    print("Did we succeed? =>", use_test_label == classification)