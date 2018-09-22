import tensorflow as tf
import tensorflow.contrib.slim as slim
#import tensorflow.nn as slim
import numpy as np
from A3CAgent.helpers import *

class AC_Network():
    def __init__(self,s_size,a_size,scope,trainer,s_shape):
        with tf.variable_scope(scope):
            #Input and visual encoding layers
            self.inputs = tf.placeholder(shape=[None,s_size],dtype=tf.float32) # input layer 
            self.imageIn = tf.reshape(self.inputs,shape=[-1,s_shape[0],s_shape[1],s_shape[2]]) # reshape
            self.conv1 = slim.conv2d(
                activation_fn=tf.nn.elu,
                inputs=self.imageIn,
                num_outputs=16,
                kernel_size=[8,8],
                stride=[4,4],
                padding='VALID') # convolution layer 1
            self.conv2 = slim.conv2d(
                activation_fn=tf.nn.elu,
                inputs=self.conv1,
                num_outputs=32,
                kernel_size=[4,4],
                stride=[2,2],
                padding='VALID') # covolution layer 2
            conv_flattened = slim.flatten(
                inputs=self.conv2) # comvlution layer output flatened
            hidden = slim.fully_connected(
                inputs=conv_flattened,
                num_outputs=256,
                activation_fn=tf.nn.elu) # hiddent layer with relu acrivation function
            
            #Recurrent network for temporal dependencies
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(
                num_units=256,
                state_is_tuple=True) #lstm cell(rnn layer)
            
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            self.state_init = [c_init, h_init]
            
            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            self.state_in = (c_in, h_in)

            rnn_in = tf.expand_dims(hidden, [0]) # entry to lstm cell(rnn)
            step_size = tf.shape(self.imageIn)[:1]
            state_in = tf.nn.rnn_cell.LSTMStateTuple(c_in, h_in)

            lstm_outputs, lstm_state = tf.nn.dynamic_rnn( 
                lstm_cell, 
                rnn_in, 
                initial_state=state_in, 
                sequence_length=step_size,
                time_major=False) # lstm layer
            lstm_c, lstm_h = lstm_state
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            rnn_out = tf.reshape(lstm_outputs, [-1, 256]) #reshape lstm output 
            
            #Output layers for policy and value estimations
            self.policy = slim.fully_connected(
                inputs=rnn_out,
                num_outputs=a_size,
                activation_fn=tf.nn.softmax,
                weights_initializer=normalized_columns_initializer(0.01), #initialzing policy for small values stability
                biases_initializer=None) # policy outputlayer with softmax activation
            self.value = slim.fully_connected(
                inputs=rnn_out,
                num_outputs=1,
                activation_fn=None, #retusrns a value 
                weights_initializer=normalized_columns_initializer(1.0),
                biases_initializer=None) # value outputlayer with no activation

            #Only the worker network need ops for loss functions and gradient updating.
            if scope != 'global':
                self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
                self.actions_onehot = tf.one_hot(indices=self.actions,depth=a_size,dtype=tf.float32)
                self.target_v = tf.placeholder(shape=[None],dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None],dtype=tf.float32)
                
                #SELECTING ACTION FROM POLICY
                self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1]) #policy output selecting action with highest liklihood to gain maximum reward

                #Loss
                self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value,[-1])))
                self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy))
                self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs)*self.advantages)
                self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.01

                #Get gradients from local network using local losses
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss,local_vars)
                self.var_norms = tf.global_norm(local_vars)
                self.grads,self.grad_norms = tf.clip_by_global_norm(self.gradients,40.0)
                
                #Apply local gradients to global network
                self.global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(self.grads,self.global_vars))