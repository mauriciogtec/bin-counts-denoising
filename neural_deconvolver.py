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


class Deconvolver(tf.keras.Model):
    def __init__(self, nblocks, ksize, filters):
        super(Deconvolver, self).__init__()
        self.nblocks = nblocks
        self.ksize = ksize
        self.filters = filters
        self.blocks = []
        self.conv_0 = tf.keras.layers.Conv1D(
          filters, ksize, activation='linear', padding='same')
        for i in range(nblocks):
            new_block = ResBlock(ksize, filters)
            self.blocks.append(new_block)
        self.conv_final = tf.keras.layers.Conv1D(
            1, 1, activation='linear', padding='same')

    def call(self, inputs):
        #  exepctec input is an array of None x bins x 2
        #  first channel corresponds to bincount
        #  second channel corresponds to peak desired
        x = inputs
        x = self.conv_0(x)
        for block in self.blocks:
            x = block(x)
        x = self.conv_final(x)
        signal_inputs = tf.expand_dims(inputs[:, :, 0], -1)
        x = x + signal_inputs
        x = tf.keras.activations.relu(x)
        x += 1e-10
        x /= tf.reduce_sum(x, axis=1, keepdims=True)
        return x
