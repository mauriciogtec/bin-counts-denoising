import tensorflow as tf


class ResBlock(tf.keras.Model):
    def __init__(self, ksize, filters, first_layer=False):
        super(ResBlock, self).__init__()
        self.ksize = ksize
        self.filters = filters
        # block 1
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.Activation('relu')
        self.conv1 = tf.keras.layers.Conv1D(
            filters, 1, activation='linear', padding='same')
        # block 2
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.Activation('relu')
        self.conv2 = tf.keras.layers.Conv1D(
            self.filters // 2, ksize, activation='linear', padding='same')
        #
        self.bn2_2 = tf.keras.layers.BatchNormalization()
        self.relu2_2 = tf.keras.layers.Activation('relu')
        k2 = self.ksize // 2 + 1
        self.conv2_2 = tf.keras.layers.Conv1D(
            self.filters // 2, k2, activation='linear', padding='same')
        #
        self.concat = tf.keras.layers.Concatenate()
        # block 3
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.relu3 = tf.keras.layers.Activation('relu')
        self.conv3 = tf.keras.layers.Conv1D(
            self.filters, 1, activation='linear', padding='same')
        # addition
        self.add = tf.keras.layers.add
        self.relu4 = tf.keras.layers.Activation('relu')

    def call(self, inputs):
        x = inputs
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x_1 = self.relu2(x)
        x_1 = self.conv2(x_1)
        x_2 = self.relu2_2(x)
        x_2 = self.conv2_2(x_2)
        x = self.concat([x_1, x_2])
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
          filters, ksize, activation='linear', padding='same')
        for i in range(nblocks):
            new_block = ResBlock(ksize, filters)
            self.blocks.append(new_block)

    def call(self, inputs):
        x = inputs
        x = self.conv_0(x)
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
            1, 1, activation='linear', padding='same')
        self.conv_final_peaks = tf.keras.layers.Conv1D(
            1, 1, activation='linear', padding='same')

    def call(self, inputs):
        x, _ = inputs
        obs = tf.expand_dims(inputs[1][:, :, 0], -1)
        # signal
        x1 = x
        for block in self.blocks_signal:
            x1 = block(x1)
        x1 = self.conv_final_signal(x1)
        x1 += obs
        x1 = tf.keras.activations.relu(x1)
        x1 += 1e-10
        x1 /= tf.reduce_sum(x1, axis=1, keepdims=True)
        signal = x1
        # peaks
        x2 = x
        for block in self.blocks_peaks:
            x2 = block(x2)
        x2 = self.conv_final_peaks(x2)
        x2 += obs
        x2 = tf.clip_by_value(x2, -10.0, 10.0)
        x2 = tf.math.sigmoid(x2)
        peaks = x2
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
            1, 1, activation='linear', padding='same')

    def call(self, inputs):
        x, _ = inputs
        obs = tf.expand_dims(inputs[1][:, :, 0], -1)
        # signal
        x1 = x
        for block in self.blocks:
            x1 = block(x1)
        x1 = self.conv_final_signal(x1)
        x1 += obs
        x1 = tf.keras.activations.relu(x1)
        x1 += 1e-10
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


