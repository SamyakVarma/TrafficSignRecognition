import keras
import numpy as np
import theano.tensor as T
import tensorflow as tf

np.bool=np.bool_

class TrafficSignNet:
    def build(width,height,depth,classes):
        inputShape=(height,width,depth)
        chanDim=-1
        model=keras.models.Sequential(
            layers=[
            keras.layers.Input((48,48,3)),
            keras.layers.Normalization(mean=[0.4914, 0.4822, 0.4465], 
                  variance=[np.square(0.247), 
                            np.square(0.243), 
                            np.square(0.261)]),
            keras.layers.Conv2D(filters=200, kernel_size=(7,7), padding='valid', activation='relu', kernel_initializer="he_normal"),
            keras.layers.MaxPool2D(pool_size=(2, 2)),
            LRN2D(),
            keras.layers.Conv2D(filters=250, kernel_size=(4,4), padding='valid', activation='relu', kernel_initializer="he_normal"),
            keras.layers.MaxPool2D(pool_size=(2, 2)),
            LRN2D(),
            keras.layers.Dropout(0.25),
            keras.layers.Conv2D(filters=350, kernel_size=(4,4), padding='valid', activation='relu', kernel_initializer="he_normal"),
            keras.layers.MaxPool2D(pool_size=(2, 2)),
            LRN2D(),
            keras.layers.Dropout(0.25),
            keras.layers.Flatten(),
            keras.layers.Dense(400,activation="relu"),
            keras.layers.Dense(classes,activation="softmax"),
            ]
        )

        return model
    
class LRN2D(keras.layers.Layer):
    """
    Local Contrast Normalisation
    This code is adapted from pylearn2.
    License at: https://github.com/lisa-lab/pylearn2/blob/master/LICENSE.txt
    """

    def __init__(self, alpha=1e-4, k=2, beta=0.75, n=5):
        if n % 2 == 0:
            raise NotImplementedError("LRN2D only works with odd n. n provided: " + str(n))
        super(LRN2D, self).__init__()
        self.alpha = alpha
        self.k = k
        self.beta = beta
        self.n = n

    def get_output(self, train):
        X = self.get_input(train)
        b, ch, r, c = X.shape
        half_n = self.n // 2
        input_sqr = tf.math.square(X)
        extra_channels = T.alloc(0., b, ch + 2*half_n, r, c)
        input_sqr = T.set_subtensor(extra_channels[:, half_n:half_n+ch, :, :], input_sqr)
        scale = self.k
        for i in range(self.n):
            scale += self.alpha * input_sqr[:, i:i+ch, :, :]
        scale = scale ** self.beta
        return X / scale

    def get_config(self):
        return {"name": self.__class__.__name__,
                "alpha": self.alpha,
                "k": self.k,
                "beta": self.beta,
                "n": self.n}
    
class CosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, min_lr, max_lr, warmup_steps=4000, alpha=0):
        super(CosineDecay, self).__init__()

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.warmup_steps = warmup_steps
        self.max_steps = warmup_steps * 10
        self.alpha = alpha

    def __call__(self, step):
        arg1 = (self.max_lr - self.min_lr) * step / self.warmup_steps + self.min_lr
        min_step = tf.math.minimum(step, self.max_steps)
        cosine_decay = 0.5 * (1 + tf.math.cos(np.pi * min_step / self.max_steps))
        decayed = (1 - self.alpha) * cosine_decay + self.alpha
        arg2 = (self.max_lr - self.min_lr) * decayed + self.min_lr
        return tf.math.minimum(arg1, arg2)
    
    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        # Get the current learning rate from model's optimizer.
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        print("\nEpoch %05d: Current learning rate is %6.4f." % (epoch, lr))
        