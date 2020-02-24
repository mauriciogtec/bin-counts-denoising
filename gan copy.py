import tensorflow as tf
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.initializers import TruncatedNormal


class ResBlock(tf.keras.Model):
    def __init__(self,
                 ksize,
                 filters,
                 leaky=False,
                 pooling=True,
                 noisy=False):
        super(ResBlock, self).__init__()
        self.ksize = ksize
        self.filters = filters
        self.pooling = pooling
        self.noisy = noisy
        self.leaky = leaky
        # block 1
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv1 = tf.keras.layers.Conv1D(
            filters, 1, activation='linear', padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(0.0001),
            bias_regularizer=tf.keras.regularizers.l2(0.00001),
            kernel_initializer=TruncatedNormal(stddev=0.02))
        # block 2
        self.bn2 = tf.keras.layers.BatchNormalization()
        #
        self.conv2_1 = tf.keras.layers.Conv1D(
            self.filters // 4, ksize, activation='linear', padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(0.0001),
            bias_regularizer=tf.keras.regularizers.l2(0.00001),
            kernel_initializer=TruncatedNormal(stddev=0.02))
        #
        k2 = max(self.ksize - 2, 1)
        self.conv2_2 = tf.keras.layers.Conv1D(
            self.filters // 4, k2, activation='linear', padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(0.0001),
            bias_regularizer=tf.keras.regularizers.l2(0.00001),
            kernel_initializer=TruncatedNormal(stddev=0.02))
        #
        k2 = max(self.ksize - 4, 1)
        self.conv2_3 = tf.keras.layers.Conv1D(
            self.filters // 4, k2, activation='linear', padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(0.0001),
            bias_regularizer=tf.keras.regularizers.l2(0.00001),
            kernel_initializer=TruncatedNormal(stddev=0.02))
        #
        # k2 = max(self.ksize - 6, 1)
        self.conv2_4 = tf.keras.layers.Conv1D(
            self.filters // 4, 1, activation='linear', padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(0.0001),
            bias_regularizer=tf.keras.regularizers.l2(0.00001),
            kernel_initializer=TruncatedNormal(stddev=0.02))
        self.maxpool = tf.keras.layers.MaxPooling1D(
            pool_size=self.ksize - 2, strides=1, padding='same')
        #
        self.concat = tf.keras.layers.Concatenate()
        # block 3
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv1D(
            self.filters, 1, activation='linear', padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(0.0001),
            bias_regularizer=tf.keras.regularizers.l2(0.00001),
            kernel_initializer=TruncatedNormal(stddev=0.02))
        # addition
        self.add = tf.keras.layers.add
        self.relu4 = tf.nn.leaky_relu

    def call(self, inputs):
        x = inputs
        x = self.bn1(x)
        if self.leaky:
            x = tf.nn.leaky_relu(x, alpha=0.1)
        else:
            x = tf.nn.relu(x)
        if self.noisy:
            x += tf.random.gamma(tf.shape(x), alpha=1.0, beta=1e3)
        x = self.conv1(x)
        x = self.bn2(x)
        if self.leaky:
            x = tf.nn.leaky_relu(x, alpha=0.1)
        else:
            x = tf.nn.relu(x)
        if self.noisy:
            x += tf.random.gamma(tf.shape(x), alpha=1.0, beta=1e3)
        x_1 = self.conv2_1(x)
        x_2 = self.conv2_2(x)
        x_3 = self.conv2_3(x)
        x_4 = self.conv2_4(x)
        if self.pooling:
            x_4_1 = self.maxpool(x_4)
            x_4 = x_4 + x_4_1
        x = self.concat([x_1, x_2, x_3, x_4])
        x = self.bn3(x)
        if self.leaky:
            x = tf.nn.leaky_relu(x, alpha=0.1)
        else:
            x = tf.nn.relu(x)
        if self.noisy:
            x += tf.random.gamma(tf.shape(x), alpha=1.0, beta=1e3)
        x = self.conv3(x)
        x = self.add([x, inputs])
        if self.leaky:
            x = tf.nn.leaky_relu(x, alpha=0.1)
        else:
            x = tf.nn.relu(x)
        if self.noisy:
            x += tf.random.gamma(tf.shape(x), alpha=1.0, beta=1e3)
        return x


class Generator(tf.keras.Model):
    def __init__(self, ksize, filters, nblocks):
        super(Generator, self).__init__()
        self.nblocks = nblocks
        self.ksize = ksize
        self.filters = filters
        self.blocks = []

        self.conv_0 = tf.keras.layers.Conv1D(
            filters, ksize, activation='linear', padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(0.0001),
            bias_regularizer=tf.keras.regularizers.l2(0.00001),
            kernel_initializer=TruncatedNormal(stddev=0.02))
        for i in range(nblocks):
            new_block = ResBlock(
                ksize, filters, pooling=False,
                leaky=False, noisy=True)
            self.blocks.append(new_block)
        self.gate_block = ResBlock(
            ksize, filters, pooling=False, noisy=True)
        self.gate_head = tf.keras.layers.Conv1D(
            1, 1, activation='linear', padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
            activity_regularizer=tf.keras.regularizers.l2(0.1),
            bias_regularizer=tf.keras.regularizers.l2(0.0001),
            kernel_initializer=TruncatedNormal(stddev=0.02),
            bias_initializer='zeros')
        self.noise_block = ResBlock(
            ksize, filters, pooling=False, noisy=True)
        self.noise_head = tf.keras.layers.Conv1D(
            1, 1, activation='linear', padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(0.0001),
            bias_regularizer=tf.keras.regularizers.l2(0.00001),
            kernel_initializer=TruncatedNormal(stddev=0.02),
            bias_initializer='zeros')

    def call(self, inputs):
        noise, signal, nr = inputs
        fnoise = tf.tile(nr, (1, tf.shape(noise)[1], 1))
        x = tf.concat([noise, fnoise, signal], -1)
        x = self.conv_0(x)
        for block in self.blocks:
            x = block(x)
        # dispersion mechanism
        x1 = x
        x1 = self.gate_block(x1)
        gate = self.gate_head(x1)
        gate = tf.clip_by_value(gate, -5.0, 5.0)
        gate += tf.random.normal(tf.shape(gate), mean=0.0, stddev=1e-3)
        x1 = signal * tf.math.exp(gate)
        x1 += tf.random.gamma(tf.shape(x1), alpha=1.0, beta=1e6)
        x1 /= tf.reduce_sum(x1, axis=1, keepdims=True)
        x_signal = x1
        # additive noise
        x2 = x
        x2 = self.noise_block(x2)
        x2 = self.noise_head(x2)
        x2 += noise
        x2 = tf.clip_by_value(x2, 1e-10, 1000.0)  # capped relu
        x2 += tf.random.gamma(tf.shape(x1), alpha=1.0, beta=1e6)
        x2 /= tf.reduce_sum(x2, axis=1, keepdims=True)
        x_noise = x2
        # extra noise
        x = nr * x_noise + (1.0 - nr) * x_signal
        return x



class Discriminator(tf.keras.Model):
    def __init__(self, ksize, filters, nblocks):
        super(Discriminator, self).__init__()
        self.nblocks = nblocks
        self.ksize = ksize
        self.filters = filters
        self.blocks = []
        for i in range(nblocks):
            block = ResBlock(
                ksize, filters,
                pooling=True, noisy=False, leaky=True)
            self.blocks.append(block)
        self.conv_0 = tf.keras.layers.Conv1D(
            filters, ksize, activation='linear', padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(0.0001),
            bias_regularizer=tf.keras.regularizers.l2(0.00001))
        self.global_pooling = tf.keras.layers.GlobalAveragePooling1D()
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.dense_final = tf.keras.layers.Dense(
            units=1,
            kernel_initializer=TruncatedNormal(stddev=0.02),
            kernel_regularizer=tf.keras.regularizers.l2(0.01),
            bias_regularizer=tf.keras.regularizers.l2(0.001),
            activity_regularizer=tf.keras.regularizers.l2(1.0),
            activation='linear')

    def call(self, inputs):
        x = inputs
        x = self.conv_0(x)
        for block in self.blocks:
            x = block(x)
        x = self.global_pooling(x)
        x = self.dropout(x)
        x = self.dense_final(x)
        return x
