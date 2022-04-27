#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 23:01:12 2022

@author: xuyierjing
"""
# In[] Setup
#import random
import gc
import numpy as np
from numpy import load, save

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.regularizers import l1,l2

import matplotlib.pyplot as plt

# In[] Create pairs of images
def make_pairs(x, y):
    """Creates a tuple containing image pairs with corresponding label.
        The pairs are created by pairing the MCI segments with each other and with the rest
    Arguments:
        x: List containing images, each index in this list corresponds to one image.
        y: List containing labels, each label with datatype of `int`.

    Returns:
        Tuple containing two numpy arrays as (pairs_of_samples, labels),
        where pairs_of_samples' shape is (2len(x), 2,n_features_dims) and
        labels are a binary array of shape (2len(x)).
    """
    pairs = []
    labels = []
    data_idx = []
    
    proto_idx = np.squeeze(np.where(y == siteA_mci))
    proto_idx_update = proto_idx
              
    for i, idx in enumerate(proto_idx):
        x1 = x[idx]
        if i != (len(proto_idx)-1):
            proto_idx_update = proto_idx_update[proto_idx_update != idx]
            for j, idx_m  in enumerate(proto_idx_update): # add a matching example
                x2 = x[idx_m]
                pairs += [[x1, x2]]
                labels += [1]
                data_idx += [[idx,idx_m]]
        for idx_nm in range(len(x)):   # add a non-matching example      
            if idx_nm not in proto_idx:
                x2 = x[idx_nm]
                pairs += [[x1, x2]]
                labels += [0]
                data_idx += [[idx,idx_nm]]

    return np.array(pairs), np.array(labels).astype("float32"), np.array(data_idx).astype("float32")

# In[] parameters
epochs = 6
batch_size = 160
margin = 1  # Margin for constrastive loss.
photo_size = 100
channel = 4
siteA_mci = 2 # the prototype

# In[] distance between embeddings
# Provided two tensors t1 and t2
# Euclidean distance = sqrt(sum(square(t1-t2)))
def euclidean_distance(vects):
    """Find the Euclidean distance between two vectors.

    Arguments:
        vects: List containing two tensors of same length.

    Returns:
        Tensor containing euclidean distance
        (as floating point value) between vectors.
    """

    x, y = vects
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))

# In[] Define the model
def create_model():
    ##################### HT's CNN version
    input = layers.Input((photo_size, photo_size, channel))
    x = layers.Conv2D(32, (3, 3), activity_regularizer=l2(0.001), activation = 'relu')(input)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.7)(x)
    x = layers.Conv2D(64, (3, 3), activation = 'relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.7)(x)
    x = layers.Conv2D(128, (3, 3), activation = 'relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.7)(x)
    x = layers.Flatten()(x)
    
    x = layers.Dense(512, activation = 'relu', kernel_initializer='glorot_uniform')(x)
    embedding_network = keras.Model(input, x)
    
    input_1 = layers.Input((photo_size, photo_size, channel))
    input_2 = layers.Input((photo_size, photo_size, channel))
    
    # As mentioned above, Siamese Network share weights between
    # tower networks (sister networks). To allow this, we will use
    # same embedding network for both tower networks.
    tower_1 = embedding_network(input_1)
    tower_2 = embedding_network(input_2)
    
    merge_layer = layers.Lambda(euclidean_distance)([tower_1, tower_2])
    ######## use cosine similarity istead of euclidean distance
    #merge_layer = layers.Dot(axes=-1, normalize=True)([tower_1, tower_2])
    normal_layer = layers.BatchNormalization()(merge_layer)
    model = keras.Model(inputs=[input_1, input_2], outputs=normal_layer)
    #output_layer = layers.Dense(1, activation="sigmoid")(normal_layer)
    #siamese = keras.Model(inputs=[input_1, input_2], outputs=output_layer)
    return model

# In[] Define the constrastive Loss

def loss(margin=1):
    """Provides 'constrastive_loss' an enclosing scope with variable 'margin'.

  Arguments:
      margin: Integer, defines the baseline for distance for which pairs
              should be classified as dissimilar. - (default is 1).

  Returns:
      'constrastive_loss' function with data ('margin') attached.
  """

    # Contrastive loss = mean( (1-true_value) * square(prediction) +
    #                         true_value * square( max(margin-prediction, 0) ))
    def contrastive_loss(y_true, y_pred):
        """Calculates the constrastive loss.

      Arguments:
          y_true: List of labels, each label is of type float32.
          y_pred: List of predictions of same length as of y_true,
                  each label is of type float32.

      Returns:
          A tensor containing constrastive loss as floating point value.
      """

        square_pred = tf.math.square(y_pred)
        margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
        return tf.math.reduce_mean(
            (1 - y_true) * square_pred + (y_true) * margin_square
        )

    return contrastive_loss

# In[] load the data
filename  = '/raid/xuyierjing/biomag_competition/training/data_SMH/photos_yoko_meanpower_sep4.npy'
labelname = '/raid/xuyierjing/biomag_competition/training/data_SMH/labels_yoko_meanpower_sep4.npy'
photos_train = load( filename ) 
labels_train = load( labelname )  

x_train = np.concatenate( (photos_train[[5,6,7,8]]), axis=3)
y_train = labels_train.ravel().astype(int)

filename  = '/raid/xuyierjing/biomag_competition/testing/testing_data/data_yoko_sep4.npy'
labelname = '/raid/xuyierjing/biomag_competition/testing/testing_data/labels_yoko_sep4.npy'
photos_test = load( filename ) 
labels_test = load( labelname )  

x_test = np.concatenate( (photos_test[[5,6,7,8]]), axis=3)
y_test = labels_test.ravel().astype(int)
  
x_add = x_train[np.squeeze(np.where(y_train == siteA_mci))]
x_test_all = np.concatenate((x_test, x_add),axis=0)

y_test_all = np.append(y_test,np.full(len(x_add), siteA_mci))

# make pairs        
pairs_train, labels_train, idx_train = make_pairs(x_train, y_train)
pairs_test, labels_test, idx_test = make_pairs(x_test_all, y_test_all)

x_train_1 = pairs_train[:, 0]
x_train_2 = pairs_train[:, 1]
x_test_1 = pairs_test[:, 0]
x_test_2 = pairs_test[:, 1] 

record = list()
for rep in range(100):
    # Train the model   
    # Clear out any old model state.
    gc.collect()
    tf.keras.backend.clear_session()
    
    # call and compile the model
    siamese = create_model()
    siamese.compile(loss=loss(margin=margin), optimizer="SGD", metrics=["accuracy"])
    #siamese.summary()
    
    # training
    history = siamese.fit(
        [x_train_1, x_train_2],
        labels_train,
        batch_size=batch_size,
        epochs=epochs,
    )

    predictions = siamese.predict([x_test_1, x_test_2])
    record.append( np.squeeze(predictions) )

    # # plot the distance
    # bins = np.linspace(-0.65, -0.35, 400)
    # #bins = np.linspace(-0.4, -0.2, 400)
    # plt.hist(predictions[np.squeeze(np.where(labels_test == 1))], bins=bins, alpha=0.5, label="SiteA_mci prototype")
    # plt.hist(predictions[np.squeeze(np.where(labels_test != 1))], bins=bins, alpha=0.5, label="Test data")
    # plt.xlabel("Data distance", size=14)
    # plt.ylabel("Count", size=14)
    # #plt.title("Data distance")
    # plt.legend(loc='upper right')
    # plt.show()
    # plt.savefig(f"Siamese_sep4_test_yoko{rep}.png")

np.save('Siamese_sep4_test_yoko_result', record)

labels_all = np.concatenate([np.expand_dims(labels_test, axis=1), idx_test], axis = 1)
np.save('Siamese_sep4_test_yoko_labels', labels_all)


