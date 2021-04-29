#!/usr/bin/env python
"""NoisyAnd: Multiple instance learning aggregation layer.

Multiple instance learning is a type of machine learning where instances are
grouped into "bags" and only labelled at the bag-level. This file implements the
Noisy-And aggregation layer proposed by the following paper.

  Kraus, Oren Z., Jimmy Lei Ba, and Brendan J. Frey. "Classifying and segmenting
  microscopy images with deep multiple instance learning." Bioinformatics 32.12
  (2016): i52-i59.
"""
__author__  = "Dan Grahn"
__version__ = "1.0.0"
__email__   = "dan.grahn@wright.edu"

import tensorflow as tf

class NoisyAnd(tf.keras.layers.Layer):
    """NoisyAnd multiple instance learning aggregation layer."""
    
    def __init__(self, output_dim, **kwargs):
        """Creates a new layer.

        Args:
            output_dim (int): Number of output dimensions, typically N_classes.
        """
        self.output_dim = output_dim
        super().__init__(**kwargs)
    
    def build(self, input_shape):
        self.a = 10 # Slope of activation.
        self.b = self.add_weight(
            name        = 'b',
            shape       = (1, input_shape[2]),
            initializer = 'uniform',
            trainable   = True,
        )
        super().build(input_shape)
    
    def call(self, x):
        mean  = tf.math.reduce_mean(x, axis=[1])
        numer = tf.math.sigmoid(self.a * (mean - self.b)) - tf.math.sigmoid(-self.a * self.b)
        denom = tf.math.sigmoid(self.a * (1 - self.b)) - tf.math.sigmoid(-self.a * self.b)
        return tf.clip_by_value(numer / denom, 0, 1)
    
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[3]
    
