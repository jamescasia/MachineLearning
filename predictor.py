import tensorflow as tf
setosa = [ [5.1, 3.5, 1.4,0.2],[5,3.3,1.4,0.2]]
versicolor = [[7,3.2,4.7,1.4], [5.7,2.8,4.1,1.3]]
virginica = [[6.3,3.3,6,2.5] , [5.9,3,5.1,1.8]]
flours = [[5.1, 3.5, 1.4,0.2],[5,3.3,1.4,0.2],[7,3.2,4.7,1.4], [5.7,2.8,4.1,1.3]
, [6.3,3.3,6,2.5] , [5.9,3,5.1,1.8]]
class IrisWizard:

    def __init__(self, name):
        print("iniized")

        with tf.Session() as sess:  
            x = tf.placeholder(tf.float32, [None, 4])
            
            W = tf.Variable( tf.zeros([4,3]))
            b = tf.Variable(tf.zeros([3]))
            y = tf.nn.softmax(tf.matmul(x, W) + b)
            self.x = x
            self.W = W
            self.b =b 
            self.y = y
            self.sess = sess
            tf.train.Saver().restore(sess, "C:\\Users\\James\\Desktop\\tensorflow\\models\\iris.ckpt")
            print("v1 : %s" % W.eval())
            # print("v2 : %s" % x.eval())
            print("v2 : %s" % b.eval())
            for a in flours:
                print(a)
                self.predictors(a[0] , a[1], a[2], a[3])
        # y = tf.nn.softmax(tf.matmul(x, W) + b)  
            
    def predictors(self, a,b,c,d):
        print('predictin eh')
        # for a in flours:
        i = [[a,b,c,d]]
        prediction = tf.argmax(self.y, 1)
            
        result = self.sess.run(prediction, feed_dict={self.x: i})
        print (result)
        if result == [0]:
            print ("setosa")
        elif result == [1]:
            print( "versicolor")
        elif result ==[2]:
            print ("virginica")

        pass
a = IrisWizard("momoy")
# 5.1	3.5	1.4	0.2	
# restore()
 