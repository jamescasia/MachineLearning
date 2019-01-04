import tensorflow as tf
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers


def trainModel():
    # irisfile = open("\datasets\Iris.csv", "r")
    irisfile = open("C:\\Users\\James\\Desktop\\tensorflow\\datasets\\Iris.csv", "r")
    iris_data = pd.read_csv(irisfile)
    # print(iris_data.head())
    # encode into one hot 
    iris_data = pd.get_dummies(iris_data)
    # first separate the datasets into training and testing  
    # frac is the percentage desired of the test data randomstate is random sees
    test_data = iris_data.sample(frac = 0.2, random_state = 566)
    # drop drops all the indexes of the test data so you're left with train data
    train_data = iris_data.drop(test_data.index) 

    # separate the features and the labels for both the training and the testing set.
    # filter is used to filter out columns it returns the columns
    # drop is used to remove the rows, it returns the dataset without the specific index rows
    # inputs are basically features andl labels are labels
    input_train = train_data.filter([
                                        'SepalLengthCm',
                                        'SepalWidthCm',
                                        'PetalLengthCm',
                                        'PetalWidthCm',
                                        ]) 

    label_train = train_data.filter([
                                        'Species_Iris-setosa',
                                        'Species_Iris-versicolor',
                                        'Species_Iris-virginica',
                                        
                                        ])
    input_test = test_data.filter([
                                        'SepalLengthCm',
                                        'SepalWidthCm',
                                        'PetalLengthCm',
                                        'PetalWidthCm',
                                        ]) 
    label_test = test_data.filter([
                                        'Species_Iris-setosa',
                                        'Species_Iris-versicolor',
                                        'Species_Iris-virginica', 
                                        ])

    # print(input_train.head())
    # print(label_train.head())
    # print(input_test.head()) 
    # print(label_test.head())

    # now the nitty gritty part, building the model 
    # let's try using the standard linear regression 

    x = tf.placeholder(tf.float32, [None, 4])
    # the 2nd parameter means the dimensions of the input None, 4 in this 
    # instance because we have a linear test data
    # our x input is a 4 length array of the width sepal length etc 4 at a time
    W = tf.Variable( tf.zeros([4,3]))
    b = tf.Variable(tf.zeros([3]))

    y = tf.nn.softmax(tf.matmul(x, W) + b)

    actual = tf.placeholder(tf.float32, [None,3 ])


    lf = tf.reduce_mean(-tf.reduce_sum(
        actual * tf.log(y), reduction_indices=[1]
    ))

    model = keras.Sequential([
    tf.keras.layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_data.keys())]),
    tf.keras.layers.Dense(64, activation=tf.nn.relu),
    tf.keras.layers.Dense(1)
  ])
    print(str(model))


    lr = 0.05  # learning rate
    train_step = tf.train.GradientDescentOptimizer(lr).minimize(lf)
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    accuracies = []
    print("TRAINING MODEL ...")
    for step in range(2000):
        sess.run(train_step, feed_dict={
            x: input_train,
            actual: label_train
            })
        if step %100 == 0:
            corrects = tf.equal(tf.argmax(y, 1), tf.argmax(actual, 1))
            accuracy = tf.reduce_mean(tf.cast(corrects, tf.float32))
            accuracy = (sess.run(accuracy * 100.0, feed_dict={
                x: input_test,
                actual: label_test
            }))
            
            # print(accuracy , max(accuracies))
            
            print (str( accuracy) + "%")
            if(len(accuracies) > 0 and  accuracy > max(accuracies)):  
                saver = tf.train.Saver()
                save_path = saver.save(sess, "C:\\Users\\James\\Desktop\\tensorflow\\models\\iris.ckpt")
                print("Model saved in path: %s" % save_path, "with" , accuracy, "accuracy")
            accuracies.append(accuracy)
            if( len(accuracies) > 6 and accuracies[(step%100)-1] == accuracy and accuracies[(step%100)-2] == accuracy and accuracies[(step%100)-3] == accuracy and accuracies[(step%100)-4] == accuracy):
                
                break


trainModel()
def predict_what_flower_this_is(sl, sw, pl, pw):
    with tf.Session() as sess:  
        x = tf.placeholder(tf.float32, [None, 4])
        
        W = tf.Variable( tf.zeros([4,3]))
        b = tf.Variable(tf.zeros([3]))
        y = tf.nn.softmax(tf.matmul(x, W) + b)
        tf.train.Saver().restore(sess, "C:\\Users\\James\\Desktop\\tensorflow\\models\\iris.ckpt")
        print("v1 : %s" % W.eval())
        # print("v2 : %s" % x.eval())
        print("v2 : %s" % b.eval())
    # y = tf.nn.softmax(tf.matmul(x, W) + b) 
        y = tf.nn.softmax(tf.matmul(x, W) + b)
        i = [[sl, sw, pl, pw]]
        prediction = tf.argmax(y, 1)
    
        result = sess.run(prediction, feed_dict={x: i})
        print (result)
        if result == [0]:
            print ("setosa")
        elif result == [1]:
            print( "versicolor")
        elif result ==[2]:
            print ("virginica")
 
# 5.1	3.5	1.4	0.2	
# predict_what_flower_this_is(5.1, 3.5, 1.4, 0.2)
