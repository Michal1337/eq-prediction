import tensorflow as tf
from typing import List


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, d_model: int):
        super().__init__()
        self.lstm = tf.keras.layers.Dense(d_model // 9 * 6)
        self.embed_dd = tf.keras.layers.Embedding(15, d_model // 9)
        self.embed_plate = tf.keras.layers.Embedding(64, d_model // 9)
        self.emebd_magtype = tf.keras.layers.Embedding(20, d_model // 9)
        self.conc = tf.keras.layers.Concatenate()
        self.pos_encoding = tf.keras.layers.Embedding(64, d_model)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        cont, plate, dd, magtype = x[:, :, :-3], x[:, :, -3], x[:, :, -2], x[:, :, -1]
        x1 = self.lstm(cont)
        x2 = self.embed_dd(dd)
        x3 = self.embed_plate(plate)
        x4 = self.emebd_magtype(magtype)
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
    def call(self, x: tf.Tensor) -> tf.Tensor:
        attn_output = self.mha(query=x, value=x, key=x)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class CrossAttention(BaseAttention):
    def call(self, x: tf.Tensor, context: tf.Tensor) -> tf.Tensor:
        attn_output = self.mha(query=x, key=context, value=context)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model: int, dff: int, dropout_rate: float = 0.1):
        super().__init__()
        self.seq = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(dff, activation="relu"),
                tf.keras.layers.Dense(d_model),
                tf.keras.layers.Dropout(dropout_rate),
            ]
        )
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x)
        return x


class EncoderBlock(tf.keras.layers.Layer):
    def __init__(
        self, d_model: int, num_heads: int, dff: int, dropout_rate: float = 0.1
    ):
        super().__init__()

        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads, key_dim=d_model, dropout=dropout_rate
        )

        self.ffn = FeedForward(d_model, dff)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.self_attention(x)
        x = self.ffn(x)
        return x


class Encoder(tf.keras.layers.Layer):
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        dff: int,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.dense = tf.keras.layers.Dense(d_model)
        self.embed_plate = tf.keras.layers.Embedding(64, d_model)
        self.conc = tf.keras.layers.Concatenate(axis=1)
        self.enc_blocks = [
            EncoderBlock(
                d_model=d_model, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate
            )
            for _ in range(1)
        ]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.dropout(x)
        x = tf.reshape(x, (-1, 4, 1))
        cont, plate = x[:, :-1, :], x[:, -1, :]
        x1 = self.dense(cont)
        x2 = self.embed_plate(plate)
        x = self.conc([x1, x2])
        for block in self.enc_blocks:
            x = block(x)

        return x


class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self: int, d_model: int, num_heads: int, dff: int, dropout_rate: float = 0.1):
        super(DecoderBlock, self).__init__()

        self.causal_self_attention = GlobalSelfAttention(
            num_heads=num_heads, key_dim=d_model, dropout=dropout_rate
        )

        self.cross_attention = CrossAttention(
            num_heads=num_heads, key_dim=d_model, dropout=dropout_rate
        )

        self.ffn = FeedForward(d_model, dff)

    def call(self, x: tf.Tensor, context: tf.Tensor) -> tf.Tensor:
        x = self.causal_self_attention(x=x)
        x = self.cross_attention(x=x, context=context)
        x = self.ffn(x)
        return x


class Decoder(tf.keras.layers.Layer):
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        dff: int,
        dropout_rate: float = 0.1,
    ):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dec_blocks = [
            DecoderBlock(
                d_model=d_model, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate
            )
            for _ in range(num_layers)
        ]

    def call(self, x: tf.Tensor, context: tf.Tensor) -> tf.Tensor:
        x = self.pos_embedding(x)
        x = self.dropout(x)
        for block in self.dec_blocks:
            x = block(x, context)
        return x


@tf.keras.utils.register_keras_serializable()
class Transformer(tf.keras.Model):
    def __init__(
        self, num_layers: int, d_model: int, num_heads: int, dropout_rate: float = 0.1
    ):
        super().__init__()
        self.decoder = Decoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=4 * d_model,
            dropout_rate=dropout_rate,
        )

        self.encoder = Encoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=4 * d_model,
            dropout_rate=dropout_rate,
        )

        self.final_layer = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, inputs: List[tf.Tensor]) -> tf.Tensor:
        x, context = inputs
        context = self.encoder(context)
        x = self.decoder(x, context)
        x = x[:, -1, :]
        logits = self.final_layer(x)
        return logits
