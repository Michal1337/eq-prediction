import tensorflow as tf

@tf.keras.saving.register_keras_serializable()
class MyModel(tf.keras.Model):
    def __init__(self, n_embed):
        super(MyModel, self).__init__()
        self.embed_plate = tf.keras.layers.Embedding(64, n_embed // 9)
        self.embed_dd = tf.keras.layers.Embedding(20, n_embed // 9)
        self.embed_magtype = tf.keras.layers.Embedding(20, n_embed // 9)
        self.embed = tf.keras.layers.Dense(n_embed // 9 * 6)
        self.lstm1 = tf.keras.layers.LSTM(n_embed, return_sequences=True)
        self.lstm2 = tf.keras.layers.LSTM(n_embed)
        self.dense1 = tf.keras.layers.Dense(n_embed, activation='relu')
        self.dense2 = tf.keras.layers.Dense(1, activation='sigmoid')
        self.conc = tf.keras.layers.Concatenate(axis=-1)

    def call(self, x):
        cont, plate, dd, magtype = x[:,:,:-3], x[:,:,-3], x[:,:,-2], x[:,:,-1] 
        x1 = self.embed(cont)
        x2 = self.embed_plate(plate)
        x3 = self.embed_dd(dd)
        x4 = self.embed_magtype(magtype)
        x = self.conc([x1, x2, x3, x4])
        x = self.lstm1(x)
        x = self.lstm2(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x