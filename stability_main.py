import models
import cPickle
import numpy as np
import tensorflow as tf
from StabilityImageReader import Image2DReader
from stability_model import resnet
train_str = "train"

filler_str = "./filler.cfg" #give location of filler file

train_batch_size = 2 #must be small for 32 layer resnet and stability training
num_classes = 2
test_batch_size = 2

#set up image feed

train_reader = Image2DReader(train_str,filler_str,train_batch_size, num_classes)

image_input_node = train_reader.get_image_batch_node()
manip_input_node = train_reader.get_manip_batch_node()
label_input_node = train_reader.get_label_batch_node()

test_reader  = Image2DReader("test",filler_str,test_batch_size, num_classes)

image_input_node = test_reader.get_image_batch_node()
manip_input_node = test_reader.get_manip_batch_node()
label_input_node = test_reader.get_label_batch_node()


batch_size = train_batch_size

image_shape = train_reader.get_image_shape()
print "Image Batch Shape: ",image_shape

X = tf.placeholder( tf.float32, shape=[train_reader.batch_size]+image_shape, name="image_input" )
M = tf.placeholder( tf.float32, shape=[train_reader.batch_size]+image_shape, name="image_input" )
Y = tf.placeholder( tf.float32, [train_reader.batch_size,num_classes], "label_input" )


#X = tf.placeholder("float", [batch_size, 756, 864, 3])
#Y = tf.placeholder("float", [batch_size, 2])
learning_rate = tf.placeholder("float", [])

#choices for how to run the net. First input is images, second is number of layers. We have limited memory, remember that
# ResNet Models
#net = resnet(X,20)
net = resnet(X, 32)
Mnet = resnet(M, 32)
#net = models.resnet(X, 32)
# net = models.resnet(X, 44)
#net = models.resnet(X, 56)


with tf.name_scope("xent") as scope:
  cross_entropy = -tf.reduce_sum(Y*tf.log(net)) #this loss measures how correct our classification is
  stability = -tf.reduce_sum(net*tf.log(Mnet)) #this loss meausres how robust against noise our classification is
  loss_func = cross_entropy+stability # add them together. You probably want to adjust the relative weights, Multiply CE by 3?
  ce_summ = tf.scalar_summary("cross entropy", cross_entropy) #for tensorboard
  stability_err = tf.scalar_summary("stability_err", stability)
with tf.name_scope("train") as scope:
  train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_func)#learning rate is given through the feed dictionary in the loop

with tf.name_scope("test") as scope:
  correct_prediction = tf.equal(tf.argmax(Y,1), tf.argmax(net,1)) #did we classify correctly?
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  accuracy_summary = tf.scalar_summary("accuracy", accuracy) # for tensorboard

saver = tf.train.Saver()
sess = tf.Session()

merged = tf.merge_all_summaries()
writer = tf.train.SummaryWriter("/tmp/stability_logs", sess.graph_def) # where do you want to put your tensorboard logs?

image_input_node = train_reader.get_image_batch_node() #make nodes for image input
image_manip_node = train_reader.get_manip_batch_node()

label_input_node = train_reader.get_label_batch_node()

sess.run(tf.initialize_all_variables())

correct_prediction = tf.equal(tf.argmax(net, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

train_reader.startQueue(sess, train_reader.batch_size) #start queue for image feed
test_reader.startQueue(sess, test_reader.batch_size)


checkpoint = "stability_bigstep-488000" # comment these three lines out if you arent uploading your own weights
print "Restoring from checkpoint", checkpoint
saver.restore(sess, checkpoint)


if True:  #sorry the formatting is bad. Leave this true
    rate = 0.0005 #this is your Adam learning rate
    val = 0 # I need this number later
    for i in range (0, 50000): #how many iterations do you want? 50000 is about 12 hours

        X_in, Y_in, manip = sess.run( [train_reader.get_image_batch_node(),train_reader.get_label_batch_node(), train_reader.get_manip_batch_node()] ) # give image, noisy image and label
        if i%2000==0:
            rate = 0.98*rate #make learning rate decay over time
        feed_dict={
            X: X_in, 
            Y: Y_in,
            M: manip,
            learning_rate: rate} #load relevant information into the feed dict
        result = sess.run([merged, accuracy, train_step], feed_dict=feed_dict) #merged prepares for  tensorboard, accuracy returns accuracy and trainstep updates the weights
        summary_str = result[0] #parse the output so it is useful
        acc = result[1]
        writer.add_summary(summary_str, i) #writes to logs
        
    
        val = (val*9+acc)/float(10)
        if i%100==0:#adjust number for how often to print
            print("My_val at step %s: %s" % (i, val)) #so with batch size of 2, accuracy is useless so this is my proxy for a running average

        if i % 5000 == 0: #how often do you want to save?
            print "training on image #%d" % i
            saver.save(sess, 'stability_net', global_step=i)#the number where i is get appended to the string you give it
        if False: #adjust this statement for how often you want to find test accuracy
            how_big = 0
            print ("testing net")
            length = 2000 #how many test images do you want?
            for i in range(length):
                X_inT, Y_inT, manipT = sess.run( [test_reader.get_image_batch_node(),test_reader.get_label_batch_node(), test_reader.get_manip_batch_node()] )
                feed_dictT={
                    X: X_inT,
                    Y: Y_inT,
                    M: manipT,
                    learning_rate: rate}
                accurac = sess.run([accuracy], feed_dict=feed_dictT) #this is all the same as before. note there is no train step
                top = len(accurac)
                size = sum(accurac)
                ave = size/float(top)
                how_big = how_big+ave #so it turns out it doesnt give me useful information easily so I needed to hack a little
            print("TESTING ACCURACY")
            print how_big/float(length) #here is the percent accuracy on the the test set
            print("TESTING ACCURACY")
saver.save(sess, 'stability_pmtweight', global_step=50000) #save progress when loop is finished
sess.close() #all done
