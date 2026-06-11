
import tensorflow as tf

def focal_loss(alpha=0.25, gamma=2.0):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        eps = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1. - eps)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        return -tf.reduce_mean(alpha * tf.pow(1. - pt, gamma) * tf.math.log(pt))
    return loss

def build_model(ts_shape, static_shape):
    ts_in = tf.keras.Input(shape=ts_shape)
    x = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=32)(ts_in, ts_in)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    st_in = tf.keras.Input(shape=static_shape)
    s = tf.keras.layers.Dense(16, activation="relu")(st_in)

    fused = tf.keras.layers.Concatenate()([x, s])
    out = tf.keras.layers.Dense(1, activation="sigmoid")(fused)

    model = tf.keras.Model([ts_in, st_in], out)
    model.compile(
        optimizer="adam",
        loss=focal_loss(),
        metrics=["accuracy", tf.keras.metrics.AUC()]
    )
    return model
