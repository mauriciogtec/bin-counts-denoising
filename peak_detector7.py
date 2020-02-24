import tensorflow as tf
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.initializers import VarianceScaling


class ResBlock(tf.keras.Model):
    def __init__(self, ksize, filters, first_layer=False):
        super(ResBlock, self).__init__()
        self.ksize = ksize
        self.filters = filters
        # block 1
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.nn.relu
        self.conv1 = tf.keras.layers.Conv1D(
            filters, 1, activation='linear', padding='same',
            kernel_initializer=TruncatedNormal(stddev=0.02),
            kernel_regularizer=tf.keras.regularizers.l2(0.0001),
            bias_regularizer=tf.keras.regularizers.l2(0.00001))
        # block 2
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.nn.relu
        #
        self.conv2_1 = tf.keras.layers.Conv1D(
            self.filters // 4, ksize, activation='linear', padding='same',
            kernel_initializer=TruncatedNormal(stddev=0.02),
            kernel_regularizer=tf.keras.regularizers.l2(0.0001),
            bias_regularizer=tf.keras.regularizers.l2(0.00001))
        #
        k2 = max(self.ksize - 2, 1)
        self.conv2_2 = tf.keras.layers.Conv1D(
            self.filters // 4, k2, activation='linear', padding='same',
            kernel_initializer=TruncatedNormal(stddev=0.02),
            kernel_regularizer=tf.keras.regularizers.l2(0.0001),
            bias_regularizer=tf.keras.regularizers.l2(0.00001))
        #
        k2 = max(self.ksize - 4, 1)
        self.conv2_3 = tf.keras.layers.Conv1D(
            self.filters // 4, k2, activation='linear', padding='same',
            kernel_initializer=TruncatedNormal(stddev=0.02),
            kernel_regularizer=tf.keras.regularizers.l2(0.0001),
            bias_regularizer=tf.keras.regularizers.l2(0.00001))
        #
        # k2 = max(self.ksize - 6, 1)
        self.conv2_4 = tf.keras.layers.Conv1D(
            self.filters // 4, 1, activation='linear', padding='same',
            kernel_initializer=TruncatedNormal(stddev=0.02),
            kernel_regularizer=tf.keras.regularizers.l2(0.0001),
            bias_regularizer=tf.keras.regularizers.l2(0.00001))
        self.maxpool = tf.keras.layers.MaxPooling1D(
            pool_size=self.ksize - 2, strides=1, padding='same')
        #
        self.concat = tf.keras.layers.Concatenate()
        # block 3
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.relu3 = tf.nn.relu
        self.conv3 = tf.keras.layers.Conv1D(
            self.filters, 1, activation='linear', padding='same',
            kernel_initializer=TruncatedNormal(stddev=0.02),
            kernel_regularizer=tf.keras.regularizers.l2(0.0001),
            bias_regularizer=tf.keras.regularizers.l2(0.00001))
        # addition
        self.add = tf.keras.layers.add
        self.relu4 = tf.nn.relu

    def call(self, inputs):
        x = inputs
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.relu2(x)
        x_1 = self.conv2_1(x)
        x_2 = self.conv2_2(x)
        x_3 = self.conv2_3(x)
        x_4 = self.conv2_4(x)
        x_4_1 = self.maxpool(x_4)
        x_4 = x_4 + x_4_1
        x = self.concat([x_1, x_2, x_3, x_4])
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.conv3(x)
        x = self.add([x, inputs])
        x = self.relu4(x)
        return x


class Features(tf.keras.Model):
    def __init__(self, ksize, filters, nblocks):
        super(Features, self).__init__()
        self.nblocks = nblocks
        self.ksize = ksize
        self.filters = filters
        self.blocks = []

        self.conv_0 = tf.keras.layers.Conv1D(
            filters // 2, ksize, activation='linear', padding='same',
            kernel_initializer=TruncatedNormal(stddev=0.02))
        for i in range(nblocks):
            new_block = ResBlock(ksize, filters)
            self.blocks.append(new_block)

    def call(self, inputs):
        x = inputs
        obs = x[:, :, 0]
        obs = tf.stack((self.filters // 2) * [obs], -1)
        x = tf.concat([self.conv_0(x), obs], -1)
        for block in self.blocks:
            x = block(x)
        return x


class SignalHead(tf.keras.Model):
    def __init__(self, ksize, filters, nblocks_signal, nblocks_peaks):
        super(SignalHead, self).__init__()
        self.nblocks_signal = nblocks_signal
        self.nblocks_peaks = nblocks_peaks
        self.ksize = ksize
        self.filters = filters
        self.blocks_signal = []
        self.blocks_peaks = []
        for i in range(nblocks_signal):
            self.blocks_signal.append(ResBlock(ksize, filters))
        for i in range(nblocks_peaks):
            self.blocks_peaks.append(ResBlock(ksize, filters))
        self.conv_final_signal = tf.keras.layers.Conv1D(
            1, 1, activation='linear', padding='same',
            kernel_initializer=TruncatedNormal(stddev=0.02),
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
            bias_regularizer=tf.keras.regularizers.l2(0.0001),
            activity_regularizer=tf.keras.regularizers.l2(0.01))
        self.conv_final_peaks = tf.keras.layers.Conv1D(
            1, 1, activation='linear', padding='same',
            kernel_initializer=TruncatedNormal(stddev=0.02),
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
            bias_regularizer=tf.keras.regularizers.l2(0.0001),
            activity_regularizer=tf.keras.regularizers.l2(0.01))

    def call(self, inputs):
        x, obs = inputs
        # obs = tf.expand_dims(inputs[1][:, :, 0], -1)
        # signal
        x1 = x
        for block in self.blocks_signal:
            x1 = block(x1)
        x1 = self.conv_final_signal(x1)
        x1 += obs
        x1 = tf.clip_by_value(x1, 1e-8, 1000.0)
        x1 /= tf.reduce_sum(x1, axis=1, keepdims=True)
        signal = x1
        # peaks
        x2 = x + obs  # add it a bit earlier
        for block in self.blocks_peaks:
            x2 = block(x2)
        peaks = self.conv_final_peaks(x2)
        #
        return signal, peaks


class DeconvHead(tf.keras.Model):
    def __init__(self, ksize, filters, nblocks):
        super(DeconvHead, self).__init__()
        self.nblocks = nblocks
        self.ksize = ksize
        self.filters = filters
        self.blocks = []
        for i in range(nblocks):
            self.blocks.append(ResBlock(ksize, filters))
        self.conv_final_signal = tf.keras.layers.Conv1D(
            1, 1, activation='linear', padding='same',
            kernel_initializer=TruncatedNormal(stddev=0.02),
            bias_initializer='zeros')

    def call(self, inputs):
        x, _ = inputs
        obs = tf.expand_dims(inputs[1][:, :, 0], -1)
        # signal
        x1 = x
        for block in self.blocks:
            x1 = block(x1)
        x1 = self.conv_final_signal(x1)
        x1 += obs
        x1 = tf.clip_by_value(x1, 1e-9, 1e4)  # capped relu
        x1 /= tf.reduce_sum(x1, axis=1, keepdims=True)
        signal = x1
        return signal


class NeuralDenoiser(tf.keras.Model):
    def __init__(self, ksize, filters, nblocks_feats,
                 nblocks_signal, nblocks_peaks, nblocks_deconv):
        super(NeuralDenoiser, self).__init__()
        self.features = Features(
            ksize, filters, nblocks_feats)
        self.signal_head = SignalHead(
            ksize, filters, nblocks_signal, nblocks_peaks)
        self.deconv_head = DeconvHead(ksize, filters, nblocks_deconv)

    def call(self, inputs):
        inputs_smoother, inputs_deconv = inputs
        # first run smoother
        x = self.features(inputs_smoother)
        signal, peaks = self.signal_head([x, inputs_smoother])
        # now deconv task
        x = self.features(inputs_deconv)
        deconv = self.deconv_head([x, inputs_deconv])
        #
        return signal, peaks, deconv


