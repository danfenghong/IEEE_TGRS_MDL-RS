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
from tf_utils import random_mini_batches, convert_to_one_hot
from tensorflow.python.framework import ops


def create_placeholders(n_x1, n_x2, n_y):
    
    isTraining = tf.placeholder_with_default(True, shape=())
    x1 = tf.placeholder(tf.float32, [None, n_x1], name = "x1")
    x2 = tf.placeholder(tf.float32, [None, n_x2], name = "x2")
    x1_full = tf.placeholder(tf.float32, [None, n_x1], name = "x1_full")
    x2_full = tf.placeholder(tf.float32, [None, n_x2], name = "x2_full")
    y = tf.placeholder(tf.float32, [None, n_y], name = "Y")

    return x1, x2, x1_full, x2_full, y, isTraining

def initialize_parameters():

    
    tf.set_random_seed(1)
     
    x1_encoder_h1 = tf.get_variable("x1_encoder_h1", [144,16], initializer = tf.contrib.layers.variance_scaling_initializer(seed = 1))
    x1_encoder_b1 = tf.get_variable("x1_encoder_b1", [16], initializer = tf.zeros_initializer())
    x2_encoder_h1 = tf.get_variable("x2_encoder_h1", [21,16], initializer = tf.contrib.layers.variance_scaling_initializer(seed = 1))
    x2_encoder_b1 = tf.get_variable("x2_encoder_b1", [16], initializer = tf.zeros_initializer())
    
    x1_encoder_h2 = tf.get_variable("x1_encoder_h2", [16,32], initializer = tf.contrib.layers.variance_scaling_initializer(seed = 1))
    x1_encoder_b2 = tf.get_variable("x1_encoder_b2", [32], initializer = tf.zeros_initializer())
    x2_encoder_h2 = tf.get_variable("x2_encoder_h2", [16,32], initializer = tf.contrib.layers.variance_scaling_initializer(seed = 1))
    x2_encoder_b2 = tf.get_variable("x2_encoder_b2", [32], initializer = tf.zeros_initializer())

    x1_encoder_h3 = tf.get_variable("x1_encoder_h3", [32,64], initializer =tf.contrib.layers.variance_scaling_initializer(seed = 1))
    x1_encoder_b3 = tf.get_variable("x1_encoder_b3", [64], initializer = tf.zeros_initializer())
    x2_encoder_h3 = tf.get_variable("x2_encoder_h3", [32,64], initializer = tf.contrib.layers.variance_scaling_initializer(seed = 1))
    x2_encoder_b3 = tf.get_variable("x2_encoder_b3", [64], initializer = tf.zeros_initializer()) 

    x1_encoder_h4 = tf.get_variable("x1_encoder_h4", [64,128], initializer =tf.contrib.layers.variance_scaling_initializer(seed = 1))
    x1_encoder_b4 = tf.get_variable("x1_encoder_b4", [128], initializer = tf.zeros_initializer())
    x2_encoder_h4 = tf.get_variable("x2_encoder_h4", [64,128], initializer = tf.contrib.layers.variance_scaling_initializer(seed = 1))
    x2_encoder_b4 = tf.get_variable("x2_encoder_b4", [128], initializer = tf.zeros_initializer()) 
    
    joint_encoder_h1 = tf.get_variable("joint_encoder_h1", [256,128], initializer = tf.contrib.layers.variance_scaling_initializer(seed = 1))
    joint_encoder_b1 = tf.get_variable("joint_encoder_b1", [128], initializer = tf.zeros_initializer())
    joint_encoder_h2 = tf.get_variable("joint_encoder_h2", [128,64], initializer = tf.contrib.layers.variance_scaling_initializer(seed = 1))
    joint_encoder_b2 = tf.get_variable("joint_encoder_b2", [64], initializer = tf.zeros_initializer())
    joint_encoder_h3 = tf.get_variable("joint_encoder_h3", [64,15], initializer = tf.contrib.layers.variance_scaling_initializer(seed = 1))
    joint_encoder_b3 = tf.get_variable("joint_encoder_b3", [15], initializer = tf.zeros_initializer())       

    x1_decoder_h1 = tf.get_variable("x1_decoder_h1", [128,64], initializer = tf.contrib.layers.variance_scaling_initializer(seed = 1))
    x1_decoder_b1 = tf.get_variable("x1_decoder_b1", [64], initializer = tf.zeros_initializer())
    x2_decoder_h1 = tf.get_variable("x2_decoder_h1", [128,64], initializer = tf.contrib.layers.variance_scaling_initializer(seed = 1))
    x2_decoder_b1 = tf.get_variable("x2_decoder_b1", [64], initializer = tf.zeros_initializer())
    
    x1_decoder_h2 = tf.get_variable("x1_decoder_h2", [64,32], initializer =tf.contrib.layers.variance_scaling_initializer(seed = 1))
    x1_decoder_b2 = tf.get_variable("x1_decoder_b2", [32], initializer = tf.zeros_initializer())
    x2_decoder_h2 = tf.get_variable("x2_decoder_h2", [64,32], initializer = tf.contrib.layers.variance_scaling_initializer(seed = 1))
    x2_decoder_b2 = tf.get_variable("x2_decoder_b2", [32], initializer = tf.zeros_initializer())
    
    x1_decoder_h3 = tf.get_variable("x1_decoder_h3", [32,16], initializer =tf.contrib.layers.variance_scaling_initializer(seed = 1))
    x1_decoder_b3 = tf.get_variable("x1_decoder_b3", [16], initializer = tf.zeros_initializer())
    x2_decoder_h3 = tf.get_variable("x2_decoder_h3", [32,16], initializer = tf.contrib.layers.variance_scaling_initializer(seed = 1))
    x2_decoder_b3 = tf.get_variable("x2_decoder_b3", [16], initializer = tf.zeros_initializer())

    x1_decoder_h4 = tf.get_variable("x1_decoder_h4", [16,144], initializer =tf.contrib.layers.variance_scaling_initializer(seed = 1))
    x1_decoder_b4 = tf.get_variable("x1_decoder_b4", [144], initializer = tf.zeros_initializer())
    x2_decoder_h4 = tf.get_variable("x2_decoder_h4", [16,21], initializer = tf.contrib.layers.variance_scaling_initializer(seed = 1))
    x2_decoder_b4 = tf.get_variable("x2_decoder_b4", [21], initializer = tf.zeros_initializer())
    
    parameters = {"x1_encoder_h1": x1_encoder_h1,
                  "x1_encoder_b1": x1_encoder_b1,
                  "x2_encoder_h1": x2_encoder_h1,
                  "x2_encoder_b1": x2_encoder_b1,
                  "x1_encoder_h2": x1_encoder_h2,
                  "x1_encoder_b2": x1_encoder_b2,
                  "x2_encoder_h2": x2_encoder_h2,
                  "x2_encoder_b2": x2_encoder_b2,
                  "x1_encoder_h3": x1_encoder_h3,
                  "x1_encoder_b3": x1_encoder_b3,
                  "x2_encoder_h3": x2_encoder_h3,
                  "x2_encoder_b3": x2_encoder_b3,
                  "x1_encoder_h4": x1_encoder_h4,
                  "x1_encoder_b4": x1_encoder_b4,
                  "x2_encoder_h4": x2_encoder_h4,
                  "x2_encoder_b4": x2_encoder_b4,
                  "joint_encoder_h1": joint_encoder_h1,
                  "joint_encoder_b1": joint_encoder_b1,
                  "joint_encoder_h2": joint_encoder_h2,
                  "joint_encoder_b2": joint_encoder_b2,
                  "joint_encoder_h3": joint_encoder_h3,
                  "joint_encoder_b3": joint_encoder_b3,
                  "x1_decoder_h1": x1_decoder_h1,
                  "x1_decoder_b1": x1_decoder_b1,
                  "x2_decoder_h1": x2_decoder_h1,
                  "x2_decoder_b1": x2_decoder_b1,
                  "x1_decoder_h2": x1_decoder_h2,
                  "x1_decoder_b2": x1_decoder_b2,
                  "x2_decoder_h2": x2_decoder_h2,
                  "x2_decoder_b2": x2_decoder_b2,
                  "x1_decoder_h3": x1_decoder_h3,
                  "x1_decoder_b3": x1_decoder_b3,
                  "x2_decoder_h3": x2_decoder_h3,
                  "x2_decoder_b3": x2_decoder_b3,
                  "x1_decoder_h4": x1_decoder_h4,
                  "x1_decoder_b4": x1_decoder_b4,
                  "x2_decoder_h4": x2_decoder_h4,
                  "x2_decoder_b4": x2_decoder_b4}

    return parameters
                  
def mynetwork(x1, x2, parameters, isTraining, momentums = 0.9, epsilons = 1e-3):

    with tf.name_scope("encoder_layer_1"):
         
         x1_encoder_layer_z1 = tf.matmul(x1, parameters['x1_encoder_h1']) + parameters['x1_encoder_b1']                                       
         x1_encoder_layer_z1_bn = tf.layers.batch_normalization(x1_encoder_layer_z1, momentum = momentums, epsilon = epsilons, training = isTraining)
         x1_encoder_layer_a1 = tf.nn.relu(x1_encoder_layer_z1_bn)
         
         x2_encoder_layer_z1 = tf.matmul(x2, parameters['x2_encoder_h1']) + parameters['x2_encoder_b1']
         x2_encoder_layer_z1_bn = tf.layers.batch_normalization(x2_encoder_layer_z1, momentum = momentums, epsilon = epsilons, training = isTraining)
         x2_encoder_layer_a1 = tf.nn.relu(x2_encoder_layer_z1_bn)
         
    with tf.name_scope("encoder_layer_2"):
         
         x1_encoder_layer_z2 = tf.matmul(x1_encoder_layer_a1, parameters['x1_encoder_h2']) + parameters['x1_encoder_b2']
         x1_encoder_layer_z2_bn = tf.layers.batch_normalization(x1_encoder_layer_z2, momentum = momentums, epsilon = epsilons, training = isTraining)                                             
         x1_encoder_layer_a2 = tf.nn.relu(x1_encoder_layer_z2_bn)
         
         x2_encoder_layer_z2 = tf.matmul(x2_encoder_layer_a1, parameters['x2_encoder_h2']) + parameters['x2_encoder_b2']
         x2_encoder_layer_z2_bn = tf.layers.batch_normalization(x2_encoder_layer_z2, momentum = momentums, epsilon = epsilons, training = isTraining)                                               
         x2_encoder_layer_a2 = tf.nn.relu(x2_encoder_layer_z2_bn)

    with tf.name_scope("encoder_layer_3"):
         
         x1_encoder_layer_z3 = tf.matmul(x1_encoder_layer_a2, parameters['x1_encoder_h3']) + parameters['x1_encoder_b3']
         x1_encoder_layer_z3_bn = tf.layers.batch_normalization(x1_encoder_layer_z3, momentum = momentums, epsilon = epsilons, training = isTraining)                                             
         x1_encoder_layer_a3 = tf.nn.relu(x1_encoder_layer_z3_bn)
         
         x2_encoder_layer_z3 = tf.matmul(x2_encoder_layer_a2, parameters['x2_encoder_h3']) + parameters['x2_encoder_b3']
         x2_encoder_layer_z3_bn = tf.layers.batch_normalization(x2_encoder_layer_z3, momentum = momentums, epsilon = epsilons, training = isTraining)                                               
         x2_encoder_layer_a3 = tf.nn.relu(x2_encoder_layer_z3_bn)

    with tf.name_scope("encoder_layer_4"):
         
         x1_encoder_layer_z4 = tf.matmul(x1_encoder_layer_a3, parameters['x1_encoder_h4']) + parameters['x1_encoder_b4']
         x1_encoder_layer_z4_bn = tf.layers.batch_normalization(x1_encoder_layer_z4, momentum = momentums, epsilon = epsilons, training = isTraining)                                             
         x1_encoder_layer_a4 = tf.nn.relu(x1_encoder_layer_z4_bn)
         
         x2_encoder_layer_z4 = tf.matmul(x2_encoder_layer_a3, parameters['x2_encoder_h4']) + parameters['x2_encoder_b4']
         x2_encoder_layer_z4_bn = tf.layers.batch_normalization(x2_encoder_layer_z4, momentum = momentums, epsilon = epsilons, training = isTraining)                                               
         x2_encoder_layer_a4 = tf.nn.relu(x2_encoder_layer_z4_bn)
         
    # joint layer
    joint_encoder_layer = tf.concat([x1_encoder_layer_a4, x2_encoder_layer_a4], 1)
    
    with tf.name_scope("joint_encoder_layer"):
         
         joint_encoder_layer_z1 = tf.matmul(joint_encoder_layer, parameters['joint_encoder_h1']) + parameters['joint_encoder_b1']                                           
         joint_encoder_layer_z1_bn = tf.layers.batch_normalization(joint_encoder_layer_z1, momentum = momentums, epsilon = epsilons, training = isTraining)  
         joint_encoder_layer_a1 = tf.nn.relu(joint_encoder_layer_z1_bn)
         
         joint_encoder_layer_z2 = tf.matmul(joint_encoder_layer_a1, parameters['joint_encoder_h2']) + parameters['joint_encoder_b2']
         joint_encoder_layer_z2_bn = tf.layers.batch_normalization(joint_encoder_layer_z2, momentum = momentums, epsilon = epsilons, training = isTraining)
         joint_encoder_layer_a2 = tf.nn.relu(joint_encoder_layer_z2_bn)
         
         joint_encoder_layer_z3 = tf.matmul(joint_encoder_layer_a2, parameters['joint_encoder_h3']) + parameters['joint_encoder_b3']

    with tf.name_scope("decoder_layer_1"):
         
         x1_decoder_layer_z1 = tf.matmul(joint_encoder_layer_a1, parameters['x1_decoder_h1']) + parameters['x1_decoder_b1']                                        
         x1_decoder_layer_a1 = tf.nn.sigmoid(x1_decoder_layer_z1)
         
         x2_decoder_layer_z1 = tf.matmul(joint_encoder_layer_a1, parameters['x2_decoder_h1']) + parameters['x2_decoder_b1']
         x2_decoder_layer_a1 = tf.nn.sigmoid(x2_decoder_layer_z1)
  
    with tf.name_scope("decoder_layer_2"):
         
         x1_decoder_layer_z2 = tf.matmul(x1_decoder_layer_a1, parameters['x1_decoder_h2']) + parameters['x1_decoder_b2']                                          
         x1_decoder_layer_a2 = tf.nn.sigmoid(x1_decoder_layer_z2)
         
         x2_decoder_layer_z2 = tf.matmul(x2_decoder_layer_a1, parameters['x2_decoder_h2']) + parameters['x2_decoder_b2']                                                
         x2_decoder_layer_a2 = tf.nn.sigmoid(x2_decoder_layer_z2)

    with tf.name_scope("decoder_layer_3"):
         
         x1_decoder_layer_z3 = tf.matmul(x1_decoder_layer_a2, parameters['x1_decoder_h3']) + parameters['x1_decoder_b3']                                          
         x1_decoder_layer_a3 = tf.nn.sigmoid(x1_decoder_layer_z3)
         
         x2_decoder_layer_z3 = tf.matmul(x2_decoder_layer_a2, parameters['x2_decoder_h3']) + parameters['x2_decoder_b3']                                             
         x2_decoder_layer_a3 = tf.nn.sigmoid(x2_decoder_layer_z3)

    with tf.name_scope("decoder_layer_4"):
         
         x1_decoder_layer_z4 = tf.matmul(x1_decoder_layer_a3, parameters['x1_decoder_h4']) + parameters['x1_decoder_b4']                                            
         x1_decoder_layer_a4 = tf.nn.sigmoid(x1_decoder_layer_z4)
         
         x2_decoder_layer_z4 = tf.matmul(x2_decoder_layer_a3, parameters['x2_decoder_h4']) + parameters['x2_decoder_b4']
         x2_decoder_layer_a4 = tf.nn.sigmoid(x2_decoder_layer_z4)
         
    l2_loss =   tf.nn.l2_loss(parameters['joint_encoder_h2']) + tf.nn.l2_loss(parameters['joint_encoder_h1']) + tf.nn.l2_loss(parameters['joint_encoder_h3'])\
                + tf.nn.l2_loss(parameters['x1_encoder_h1']) + tf.nn.l2_loss(parameters['x2_encoder_h1']) + tf.nn.l2_loss(parameters['x1_encoder_h2']) + tf.nn.l2_loss(parameters['x2_encoder_h2'])\
                + tf.nn.l2_loss(parameters['x1_encoder_h3']) + tf.nn.l2_loss(parameters['x2_encoder_h3'])\
                + tf.nn.l2_loss(parameters['x2_decoder_h1']) + tf.nn.l2_loss(parameters['x1_decoder_h1'])\
                + tf.nn.l2_loss(parameters['x2_decoder_h3']) + tf.nn.l2_loss(parameters['x1_decoder_h3']) + tf.nn.l2_loss(parameters['x2_decoder_h2']) + tf.nn.l2_loss(parameters['x1_decoder_h2'])\
                + tf.nn.l2_loss(parameters['x2_decoder_h4']) + tf.nn.l2_loss(parameters['x1_decoder_h4']) + tf.nn.l2_loss(parameters['x1_encoder_h4']) + tf.nn.l2_loss(parameters['x2_encoder_h4'])
                
    return joint_encoder_layer_z3, l2_loss, x1_decoder_layer_a4, x2_decoder_layer_a4

def mynetwork_optimaization(y_es, y_re, x1_de_es, x2_de_es, x1_re, x2_re, l2_loss, reg, learning_rate, global_step):

    with tf.name_scope("cost"):
        
         cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y_es, labels = y_re)) + reg * l2_loss\
                + 1 * tf.reduce_mean(tf.pow(x1_de_es - x1_re, 2)) + 1 * tf.reduce_mean(tf.pow(x2_de_es - x2_re, 2))
               
    with tf.name_scope("optimization"):
         update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
         optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost, global_step=global_step)
         optimizer = tf.group([optimizer, update_ops])
         
    return cost, optimizer

def train_mynetwork(x1_train_set, x2_train_set, x1_train_set_full, x2_train_set_full, x1_test_set, x2_test_set, y_train_set, y_test_set, MODEL,
           learning_rate_base = 0.001, beta_reg = 0.001, num_epochs = 150, minibatch_size = 64, print_cost = True):
    
    ops.reset_default_graph()                       
    tf.set_random_seed(1)                          
    seed = 1                                    
    (m, n_x1) = x1_train_set.shape                        
    (m, n_x2) = x2_train_set.shape
    (m, n_y) = y_train_set.shape                            

    costs = []                                   
    costs_dev = []
    train_acc = []
    val_acc = []
    correct_prediction = 0
    
    # Create Placeholders of shape (n_x, n_y)
    x1, x2, x1_full, x2_full, y, isTraining = create_placeholders(n_x1, n_x2, n_y)

    # Initialize parameters
    parameters = initialize_parameters()
    
    with tf.name_scope("network"):

         joint_layer, l2_loss, x1_de, x2_de = mynetwork(x1, x2, parameters, isTraining)
         
    global_step = tf.Variable(0, trainable = False)
    learning_rate = tf.train.exponential_decay(learning_rate_base, global_step, 30 * m/minibatch_size, 0.5, staircase = True)
    
    with tf.name_scope("optimization"):
         # network optimization
         cost, optimizer = mynetwork_optimaization(joint_layer, y, x1_de, x2_de, x1_full, x2_full, l2_loss, beta_reg, learning_rate, global_step)

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
            minibatches = random_mini_full_batches(x1_train_set, x2_train_set, x1_train_set_full, x2_train_set_full, y_train_set, minibatch_size, seed)
            for minibatch in minibatches:

                # Select a minibatch
                (batch_x1, batch_x2, batch_x1_full, batch_x2_full, batch_y) = minibatch
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                _, minibatch_cost, minibatch_acc = sess.run([optimizer, cost, accuracy], feed_dict={x1: batch_x1, x2: batch_x2, x1_full: batch_x1_full, x2_full: batch_x2_full, y: batch_y, isTraining: True})
           
                epoch_cost += minibatch_cost / (num_minibatches+ 1)
                epoch_acc += minibatch_acc / (num_minibatches + 1)
 
            if MODEL.strip() == 'MML':  
               # Multimodal Learning (MML): close BN by "isTraining: False"            
               feature, epoch_cost_dev, epoch_acc_dev = sess.run([joint_layerT, cost, accuracy], feed_dict={x1: x1_test_set, x2: x2_test_set, x1_full: x1_test_set, x2_full: x2_test_set, y: y_test_set, isTraining: False})
           
            if MODEL.strip() == 'CML':  
               # Crossmodal Learning (CML): open BN by "isTraining: True"   
               feature, epoch_cost_dev, epoch_acc_dev = sess.run([joint_layerT, cost, accuracy], feed_dict={x1: x1_test_set, x2: x2_test_set, x1_full: x1_test_set, x2_full: x2_test_set, y: y_test_set, isTraining: True})
            
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
        save_path = saver.save(sess,"D:\Python_Project\MDL-RS/save_EnDe_FC/model.ckpt")
        print("save model:{0} Finished".format(save_path))
  
        return parameters, val_acc, feature

HSI_TrSet = scio.loadmat('HSI_LiDAR_FC/HSI_TrSet.mat')
LiDAR_TrSet = scio.loadmat('HSI_LiDAR_FC/LiDAR_TrSet.mat')
HSI_TeSet = scio.loadmat('HSI_LiDAR_FC/HSI_TeSet.mat')
LiDAR_TeSet = scio.loadmat('HSI_LiDAR_FC/LiDAR_TeSet.mat')

TrLabel = scio.loadmat('HSI_LiDAR_FC/TrLabel.mat')
TeLabel = scio.loadmat('HSI_LiDAR_FC/TeLabel.mat')

HSI_TrSet = HSI_TrSet['HSI_TrSet']
LiDAR_TrSet = LiDAR_TrSet['LiDAR_TrSet']
HSI_TeSet = HSI_TeSet['HSI_TeSet']
LiDAR_TeSet = LiDAR_TeSet['LiDAR_TeSet']

TrLabel = TrLabel['TrLabel']
TeLabel = TeLabel['TeLabel']

Y_train = convert_to_one_hot(TrLabel-1, 15)
Y_test = convert_to_one_hot(TeLabel-1, 15)

Y_train = Y_train.T
Y_test = Y_test.T

MODEL = 'MML' # 'MML': Multimodal learning (MML)
              # 'CML-LiDAR':  Crossmodal learning (CML-LiDAR)
              # 'CML-HSI':  Crossmodal learning (CML-HSI)

if MODEL.strip() == 'MML':
   parameters, val_acc, feature = train_mynetwork(HSI_TrSet, LiDAR_TrSet, HSI_TrSet, LiDAR_TrSet, HSI_TeSet, LiDAR_TeSet, Y_train, Y_test, 'MML')

if MODEL.strip() == 'CML-LiDAR':
   parameters, val_acc, feature = train_mynetwork(np.concatenate((HSI_TrSet, HSI_TrSet, np.zeros_like(HSI_TrSet)), 0), np.concatenate((LiDAR_TrSet, np.zeros_like(LiDAR_TrSet), LiDAR_TrSet), 0), np.concatenate((HSI_TrSet, HSI_TrSet, HSI_TrSet), 0), np.concatenate((LiDAR_TrSet, LiDAR_TrSet, LiDAR_TrSet), 0), np.zeros_like(HSI_TeSet), LiDAR_TeSet, np.concatenate((Y_train, Y_train, Y_train), 0), Y_test, 'CML')

if MODEL.strip() == 'CML-HSI':
   parameters, val_acc, feature = train_mynetwork(np.concatenate((HSI_TrSet, HSI_TrSet, np.zeros_like(HSI_TrSet)), 0), np.concatenate((LiDAR_TrSet, np.zeros_like(LiDAR_TrSet), LiDAR_TrSet), 0), np.concatenate((HSI_TrSet, HSI_TrSet, HSI_TrSet), 0), np.concatenate((LiDAR_TrSet, LiDAR_TrSet, LiDAR_TrSet), 0), HSI_TeSet, np.zeros_like(LiDAR_TeSet), np.concatenate((Y_train, Y_train, Y_train), 0), Y_test, 'CML')

sio.savemat('feature.mat', {'feature': feature})
print ("Maxmial Accuracy: %f, index: %i" % (max(val_acc),val_acc.index(max(val_acc))))
