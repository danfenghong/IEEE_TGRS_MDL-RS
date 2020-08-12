# -*- coding: utf-8 -*-
"""
Created on Fri May  4 15:46:19 2018

@author: danfeng
"""
#import library
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.io as scio 
import scipy.io as sio
from tf_utils import random_mini_batches_standard, convert_to_one_hot
from tensorflow.python.framework import ops


def create_placeholders(n_x1, n_y):

    isTraining = tf.placeholder_with_default(True, shape=())
    x1 = tf.placeholder(tf.float32, [None, n_x1], name = "x1")
    y = tf.placeholder(tf.float32, [None, n_y], name = "Y")
    
    return x1, y, isTraining

def initialize_parameters():

    
    tf.set_random_seed(1)
     
    x2_encoder_h1 = tf.get_variable("x2_encoder_h1", [21,16], initializer = tf.contrib.layers.variance_scaling_initializer(seed = 1))
    x2_encoder_b1 = tf.get_variable("x2_encoder_b1", [16], initializer = tf.zeros_initializer())
    
    x2_encoder_h2 = tf.get_variable("x2_encoder_h2", [16,32], initializer = tf.contrib.layers.variance_scaling_initializer(seed = 1))
    x2_encoder_b2 = tf.get_variable("x2_encoder_b2", [32], initializer = tf.zeros_initializer())
    
    x2_encoder_h3 = tf.get_variable("x2_encoder_h3", [32,64], initializer = tf.contrib.layers.variance_scaling_initializer(seed = 1))
    x2_encoder_b3 = tf.get_variable("x2_encoder_b3", [64], initializer = tf.zeros_initializer())

    x2_encoder_h4 = tf.get_variable("x2_encoder_h4", [64,128], initializer = tf.contrib.layers.variance_scaling_initializer(seed = 1))
    x2_encoder_b4 = tf.get_variable("x2_encoder_b4", [128], initializer = tf.zeros_initializer())   

    x2_encoder_h5 = tf.get_variable("x2_encoder_h5", [128,15], initializer = tf.contrib.layers.variance_scaling_initializer(seed = 1))
    x2_encoder_b5 = tf.get_variable("x2_encoder_b5", [15], initializer = tf.zeros_initializer())   

    parameters = {"x2_encoder_h1": x2_encoder_h1,
                  "x2_encoder_b1": x2_encoder_b1,
                  "x2_encoder_h2": x2_encoder_h2,
                  "x2_encoder_b2": x2_encoder_b2,
                  "x2_encoder_h3": x2_encoder_h3,
                  "x2_encoder_b3": x2_encoder_b3,
                  "x2_encoder_h4": x2_encoder_h4,
                  "x2_encoder_b4": x2_encoder_b4,
                  "x2_encoder_h5": x2_encoder_h5,
                  "x2_encoder_b5": x2_encoder_b5}

    return parameters


def mynetwork(x2, parameters, isTraining, momentum_v = 0.9):

    with tf.name_scope("encoder_layer_1"):

         x2_encoder_layer_z1 = tf.matmul(x2, parameters['x2_encoder_h1'])  + parameters['x2_encoder_b1']
         x2_encoder_layer_z1_bn = tf.layers.batch_normalization(x2_encoder_layer_z1, momentum = momentum_v, training = isTraining) 
         x2_encoder_layer_a1 = tf.nn.relu(x2_encoder_layer_z1_bn)
                  
    with tf.name_scope("encoder_layer_2"):
         
         x2_encoder_layer_z2 = tf.matmul(x2_encoder_layer_a1, parameters['x2_encoder_h2']) + parameters['x2_encoder_b2']
         x2_encoder_layer_z2_bn = tf.layers.batch_normalization(x2_encoder_layer_z2, momentum = momentum_v, training = isTraining)                                                  
         x2_encoder_layer_a2 = tf.nn.relu(x2_encoder_layer_z2_bn)
         
    with tf.name_scope("encoder_layer_3"):
         
         x2_encoder_layer_z3 = tf.matmul(x2_encoder_layer_a2, parameters['x2_encoder_h3']) + parameters['x2_encoder_b3']
         x2_encoder_layer_z3_bn = tf.layers.batch_normalization(x2_encoder_layer_z3, momentum = momentum_v, training = isTraining)                                                 
         x2_encoder_layer_a3 = tf.nn.relu(x2_encoder_layer_z3_bn)
         
    with tf.name_scope("encoder_layer_4"):
         
         x2_encoder_layer_z4 = tf.matmul(x2_encoder_layer_a3, parameters['x2_encoder_h4']) + parameters['x2_encoder_b4']
         x2_encoder_layer_z4_bn = tf.layers.batch_normalization(x2_encoder_layer_z4, momentum = momentum_v, training = isTraining)                                                 
         x2_encoder_layer_a4 = tf.nn.relu(x2_encoder_layer_z4_bn)
     
    with tf.name_scope("encoder_layer_5"):         
         x2_encoder_layer_z6 = tf.matmul(x2_encoder_layer_a4, parameters['x2_encoder_h5']) + parameters['x2_encoder_b5'] 
       
    l2_loss =  tf.nn.l2_loss(parameters['x2_encoder_h1']) + tf.nn.l2_loss(parameters['x2_encoder_h2'])\
               + tf.nn.l2_loss(parameters['x2_encoder_h3']) + tf.nn.l2_loss(parameters['x2_encoder_h4']) + tf.nn.l2_loss(parameters['x2_encoder_h5'])
             
    return x2_encoder_layer_z6, l2_loss

def mynetwork_optimaization(y_es, y_re, l2_loss, reg, learning_rate, global_step):

    with tf.name_scope("cost"):
        
         cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y_es, labels = y_re)) + reg * l2_loss
    
    with tf.name_scope("optimization"):
         update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
         optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost, global_step=global_step)
         optimizer = tf.group([optimizer, update_ops])
         
    return cost, optimizer

def train_mynetwork(x1_train_set, x1_test_set, y_train_set, y_test_set,
           learning_rate_base = 0.001, beta_reg = 0.01, num_epochs = 150, minibatch_size = 64, print_cost = True):
    
    ops.reset_default_graph()                       
    tf.set_random_seed(1)                          
    seed = 1                                     
    (m, n_x1) = x1_train_set.shape                        
    (m, n_y) = y_train_set.shape                            

    costs = []                                   
    costs_dev = []
    train_acc = []
    val_acc = []
    correct_prediction = 0
    
    # Create Placeholders of shape (n_x, n_y)
    x1, y, isTraining = create_placeholders(n_x1, n_y)

    # Initialize parameters
    parameters = initialize_parameters()
    
    with tf.name_scope("network"):

         joint_layer, l2_loss = mynetwork(x1, parameters, isTraining)
         
    global_step = tf.Variable(0, trainable = False)
    learning_rate = tf.train.exponential_decay(learning_rate_base, global_step, 30 * m/minibatch_size, 0.5, staircase = True)
    
    with tf.name_scope("optimization"):
         # network optimization
         cost, optimizer = mynetwork_optimaization(joint_layer, y, l2_loss, beta_reg, learning_rate, global_step)

    with tf.name_scope("metrics"):
         # Calculate the correct predictions
         joint_layerT = tf.transpose(joint_layer)
         yT = tf.transpose(y)
         correct_prediction = tf.equal(tf.argmax(joint_layerT), tf.argmax(yT))
         accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # Initialize all the variables
    init = tf.global_variables_initializer()
    saver = tf.train.Saver() 
    
    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)

        # Do the training loop
        for epoch in range(num_epochs + 1):

            epoch_cost = 0.                       # Defines a cost related to an epoch
            epoch_acc = 0.
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches_standard(x1_train_set, y_train_set, minibatch_size, seed)
            for minibatch in minibatches:

                # Select a minibatch
                (batch_x1, batch_y) = minibatch
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                _, minibatch_cost, minibatch_acc = sess.run([optimizer, cost, accuracy], feed_dict={x1: batch_x1, y: batch_y, isTraining: True})
           
                epoch_cost += minibatch_cost / (num_minibatches+ 1)
                epoch_acc += minibatch_acc / (num_minibatches + 1)
 
            feature, epoch_cost_dev, epoch_acc_dev = sess.run([joint_layerT, cost, accuracy], feed_dict={x1: x1_test_set, y: y_test_set, isTraining: False})
           
            # Print the cost every epoch
        
            if print_cost == True and epoch % 50 == 0:
                print ("epoch %i: Train_loss: %f, Val_loss: %f, Train_acc: %f, Val_acc: %f" % (epoch, epoch_cost, epoch_cost_dev, epoch_acc, epoch_acc_dev))
                
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
                train_acc.append(epoch_acc)
                costs_dev.append(epoch_cost_dev)
                val_acc.append(epoch_acc_dev)
        
        # plot the cost      
        plt.plot(np.squeeze(costs))
        plt.plot(np.squeeze(costs_dev))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
        # plot the accuracy 
        plt.plot(np.squeeze(train_acc))
        plt.plot(np.squeeze(val_acc))
        plt.ylabel('accuracy')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
        
        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")
    
        print("save model")
        save_path = saver.save(sess,"D:\Python_Project\MDL-RS/save_LiDAR/model.ckpt")
        print("save model:{0} Finished".format(save_path))
 
        return parameters, val_acc, feature


LiDAR_TrSet = scio.loadmat('HSI_LiDAR_FC/LiDAR_TrSet.mat')
LiDAR_TeSet = scio.loadmat('HSI_LiDAR_FC/LiDAR_TeSet.mat')

TrLabel = scio.loadmat('HSI_LiDAR_FC/TrLabel.mat')
TeLabel = scio.loadmat('HSI_LiDAR_FC/TeLabel.mat')

LiDAR_TrSet = LiDAR_TrSet['LiDAR_TrSet']
LiDAR_TeSet = LiDAR_TeSet['LiDAR_TeSet']

TrLabel = TrLabel['TrLabel']
TeLabel = TeLabel['TeLabel']

Y_train = convert_to_one_hot(TrLabel-1, 15)
Y_test = convert_to_one_hot(TeLabel-1, 15)

Y_train = Y_train.T
Y_test = Y_test.T

parameters, val_acc, feature = train_mynetwork(LiDAR_TrSet, LiDAR_TeSet, Y_train, Y_test)
sio.savemat('feature.mat', {'feature': feature}) # store the output
print ("Maxmial Accuracy: %f, index: %i" % (max(val_acc),val_acc.index(max(val_acc))))