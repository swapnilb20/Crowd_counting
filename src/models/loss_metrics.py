import tensorflow as tf
from keras.saving import register_keras_serializable

@register_keras_serializable()
def combined_loss(y_true, y_pred):
    y_pred = tf.maximum(y_pred, 0)
    map_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
    c_true = tf.reduce_sum(y_true, [1,2,3])
    c_pred = tf.reduce_sum(y_pred, [1,2,3])
    return 0.95*map_loss + 0.05*tf.reduce_mean(tf.abs(c_true - c_pred))

@register_keras_serializable()
def count_mae_metric(y_true, y_pred):
    y_pred = tf.maximum(y_pred, 0)
    return tf.reduce_mean(tf.abs(
        tf.reduce_sum(y_true, [1,2,3]) - tf.reduce_sum(y_pred, [1,2,3])
    ))
