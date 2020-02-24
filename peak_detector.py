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
            filters // 4, 1, activation='linear', padding='same')
        # block 2
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.Activation('relu')
        self.conv2 = tf.keras.layers.Conv1D(
            self.filters // 4, ksize, activation='linear', padding='same')
        #
        self.bn2_2 = tf.keras.layers.BatchNormalization()
        self.relu2_2 = tf.keras.layers.Activation('relu')
        k2 = self.ksize // 2 + 1
        self.conv2_2 = tf.keras.layers.Conv1D(
            self.filters // 4, k2, activation='linear', padding='same')
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


class PeakClassifier(tf.keras.Model):
    def __init__(self, nblocks, ksize, filters):
        super(PeakClassifier, self).__init__()
        self.nblocks = nblocks
        self.ksize = ksize
        self.filters = filters
        self.blocks = []
        for i in range(nblocks):
            new_block = ResBlock(ksize, filters)
            self.blocks.append(new_block)
        # self.conv_0 = tf.keras.layers.Conv1D(
        #   filters, ksize, activation='relu')
        self.peaks_layer = tf.keras.layers.Conv1D(1, 1, activation='linear')
        # self.wts_layer = tf.keras.layers.Conv1D(1, 1, activation='linear')

    def call(self, inputs):
        nbins = tf.cast(tf.shape(inputs)[1], tf.float32)
        C = tf.math.sqrt(nbins)
        x = inputs * C
        # x = self.conv_0(x)
        for block in self.blocks:
            x = block(x)
            
        #
        x = self.peaks_layer(x)
        x = tf.clip_by_value(x, -10.0, 10.0)
        x = tf.math.sigmoid(x)
        # out2 = tf.math.softmax(self.wts_layer(x), axis=1)
        return x
