import tensorflow as tf
from tensorflow.keras import datasets, layers, models

class LoraLinear(tf.keras.Model):
    def __init__(self,out,red,name='LoRA_dense'):
        super().__init__()
        self.out = out
        self.reduction = red
        self.linear = tf.keras.layers.Dense(self.out,activation='relu',use_bias=False)
        self.lora_squeeze = tf.keras.layers.Dense(self.reduction,activation='relu',use_bias=False)
        self.dropout = tf.keras.layers.Dropout(rate=0.5)
        self.lora_expand = tf.keras.layers.Dense(self.out,activation='relu',use_bias=False)
        self.scale = 1.0
        self.name=name

    def call(self,x):
        t = self.linear(x)
        x = t + self.dropout(self.lora_expand(self.lora_squeeze(x)))*self.scale
        return x