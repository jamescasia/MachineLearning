import tensorflow as tf
import pandas as pd

w = tf.Variable([-.3], tf.float32)
b = tf.Variable([.3], tf.float32)
x = tf.placeholder(tf.float32)

# model, you can modify it to anything you want 
linear = w*x +b


y = tf.placeholder(tf.float32)
squared_delta = tf.square(linear -y)

loss = tf.reduce_sum(squared_delta)

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)



init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(10000):
    sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})
# x is the training feature y is the training label
# each y is the value of model for each x
# loss = value of the model's y fed with the x's - the y's down here vv
print(sess.run([w, b] ))

File_Writer = tf.summary.FileWriter('C:\\Users\\James\\Desktop\\tensorflow\\graph', sess.graph)


# print(sess.run(c))
sess.close()