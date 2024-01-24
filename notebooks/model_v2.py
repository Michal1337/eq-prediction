import numpy as np
import tensorflow as tf
import pandas as pd
SEED = 1337
tf.random.set_seed(SEED)
np.random.seed(SEED)

ds_train = tf.data.Dataset.load("../data/ds_train/")
ds_val = tf.data.Dataset.load("../data/ds_val/")
ds_test = tf.data.Dataset.load("../data/ds_test/")

BATCH_SIZE = 1024
BLOCK_SIZE = 64

def fix_x_and_y(features, context, labels):
    return (features, context), labels

ds_train = ds_train.map(fix_x_and_y)
ds_val = ds_val.map(fix_x_and_y)
ds_test = ds_test.map(fix_x_and_y)

neg, pos = 3094474, 199842
total = neg + pos

ds_train = ds_train.batch(BATCH_SIZE).shuffle(2000, seed=SEED, reshuffle_each_iteration=False).prefetch(tf.data.experimental.AUTOTUNE)
ds_val = ds_val.batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
ds_test = ds_test.batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, d_model):
        super().__init__()
        self.dense = tf.keras.layers.Dense(d_model // 9 * 6)
        self.embed_dd = tf.keras.layers.Embedding(20, d_model // 9)
        self.embed_plate = tf.keras.layers.Embedding(64, d_model // 9)
        self.embed_magtype = tf.keras.layers.Embedding(20, d_model // 9)
        self.conc = tf.keras.layers.Concatenate()
        self.pos_encoding = tf.keras.layers.Embedding(64, d_model)

    def call(self, x):
        cont, plate, dd, magtype = x[:,:,:-3], x[:,:,-3], x[:,:,-2], x[:,:,-1] 
        x1 = self.dense(cont)
        x2 = self.embed_dd(dd)
        x3 = self.embed_plate(plate)
        x4 = self.embed_magtype(magtype)
        x = self.conc([x1, x2, x3, x4])
        x_pos = self.pos_encoding(tf.range(x1.shape[1]))
        x = x + x_pos
        return x
    
class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()

class GlobalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x
    
class CrossAttention(BaseAttention):
	def call(self, x, context):
		attn_output = self.mha(
			query=x,
			key=context,
			value=context)
		x = self.add([x, attn_output])
		x = self.layernorm(x)
		return x
     
class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dropout_rate)
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x) 
        return x
    
class EncoderBlock(tf.keras.layers.Layer):
	def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
		super().__init__()

		self.self_attention = GlobalSelfAttention(
			num_heads=num_heads,
			key_dim=d_model,
			dropout=dropout_rate)

		self.ffn = FeedForward(d_model, dff)

	def call(self, x):
		x = self.self_attention(x)
		x = self.ffn(x)
		return x
     
class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads,
               dff, dropout_rate=0.1):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.dense = tf.keras.layers.Dense(d_model)
        self.embed_plate = tf.keras.layers.Embedding(64, d_model)
        self.conc = tf.keras.layers.Concatenate(axis=1)
        self.enc_blocks = [
            EncoderBlock(d_model=d_model,
                        num_heads=num_heads,
                        dff=dff,
                        dropout_rate=dropout_rate)
            for _ in range(1)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x):
        x = self.dropout(x)
        x = tf.reshape(x, (-1, 4, 1))
        cont, plate = x[:, :-1, :], x[:, -1, :]
        x1 = self.dense(cont)
        x2 = self.embed_plate(plate)
        x = self.conc([x1, x2])
        for block in self.enc_blocks:
            x = block(x)

        return x  # Shape `(batch_size, seq_len, d_model)`.
    
class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self,
                d_model,
                num_heads,
                dff,
                dropout_rate=0.1):
        super(DecoderBlock, self).__init__()

        self.causal_self_attention = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)

        self.cross_attention = CrossAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)

        self.ffn = FeedForward(d_model, dff)

    def call(self, x, context):
        x = self.causal_self_attention(x=x)
        x = self.cross_attention(x=x, context=context)
        x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.
        return x

class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, dropout_rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dec_blocks = [
            DecoderBlock(d_model=d_model, num_heads=num_heads,
                        dff=dff, dropout_rate=dropout_rate)
            for _ in range(num_layers)]

    def call(self, x, context):
        x = self.pos_embedding(x)  # (batch_size, target_seq_len, d_model)
        x = self.dropout(x)
        for block in self.dec_blocks:
            x = block(x, context)
        # The shape of x is (batch_size, target_seq_len, d_model).
        return x
    
@tf.keras.utils.register_keras_serializable()
class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads,
                dropout_rate=0.1):
        super().__init__()
        self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
                            num_heads=num_heads, dff=4*d_model,
                            dropout_rate=dropout_rate)

        self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                           num_heads=num_heads, dff=4*d_model,
                           dropout_rate=dropout_rate)

        self.final_layer = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        x, context = inputs
        context = self.encoder(context)
        x = self.decoder(x, context) # (batch_size, target_len, d_model)
        x = x[:, -1, :]
        logits = self.final_layer(x)  # (batch_size, 1, target_vocab_size)
        return logits
    
model = Transformer(num_layers=2, d_model=72, num_heads=2)

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            optimizer='adam',
            metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])

weight_for_0 = (1 / neg) * (total / 2.0)
weight_for_1 = (1 / pos) * (total / 2.0)
class_weight = {0: weight_for_0, 1: weight_for_1}
history = model.fit(ds_train, epochs=20, validation_data=ds_val)

model.save("../models/model_v2.keras")

# save history
history = pd.DataFrame.from_dict(history.history)
history.to_csv("../models/history_model_v2.csv", index=False)