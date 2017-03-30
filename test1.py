import tensorflow as tf

node1 = tf.constant(3,name="Node1")
node2 = tf.constant(4, name="Node2")
node3 = node1+node2



print(node1,node2,node3)



sess = tf.Session()
myWriter = tf.summary.FileWriter("./myLog",sess.graph)

print(sess.run([node1]))
print(sess.run([node2]))
print(sess.run([node3]))

## to see tensorboard run command below in console
## tensorboard --logdir=./myLog
