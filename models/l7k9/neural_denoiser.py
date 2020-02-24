import tensorflow as tf


class ResBlock(tf.keras.Model):
    def __init__(self, ksize, filters):
        super(ResBlock, self).__init__()
        self.ksize = ksize
        self.filters = filters
        # block 1
        # self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.Activation('relu')
        self.conv1 = tf.keras.layers.Conv1D(
            filters // 4, 1, activation='linear', padding='same')
        # block 2
        # self.bn2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.Activation('relu')
        self.conv2 = tf.keras.layers.Conv1D(
            self.filters // 4, ksize, activation='linear', padding='same')
        # block 3
        # self.bn3 = tf.keras.layers.BatchNormalization()
        self.relu3 = tf.keras.layers.Activation('relu')
        self.conv3 = tf.keras.layers.Conv1D(
            self.filters, 1, activation='linear', padding='same')
        # addition
        self.add = tf.keras.layers.add
        self.relu4 = tf.keras.layers.Activation('relu')

    def call(self, inputs):
        x = inputs
        # x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv1(x)
        # x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv2(x)
        # x = self.bn3(x)
        x = self.relu3(x)
        x = self.conv3(x)
        x = self.add([x, inputs])
        x = self.relu4(x)
        return x


class BinDenoiser(tf.keras.Model):
    def __init__(self, nblocks, ksize, filters):
        super(BinDenoiser, self).__init__()
        self.nblocks = nblocks
        self.ksize = ksize
        self.filters = filters
        self.blocks = []
        for _ in range(nblocks):
            new_block = ResBlock(ksize, filters)
            self.blocks.append(new_block)
        self.conv_final = tf.keras.layers.Conv1D(1, 1, activation='linear')
        self.add_final = tf.keras.layers.Add()
        self.relu_final = tf.keras.layers.Activation('relu')

    def call(self, inputs):
        x = inputs
        for block in self.blocks:
            x = block(x)
        x = self.conv_final(x)
        x = self.add_final([x, inputs])
        x = self.relu_final(x)
        x += 1e-10
        x /= tf.reduce_sum(x, axis=1, keepdims=True)
        return x
