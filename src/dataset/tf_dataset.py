import os, tensorflow as tf
import numpy as np

def load_density_np(path):
    return np.load(path.decode())['density'][..., None].astype(np.float32)

def load_pair_tf(img_path, dens_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    d = tf.numpy_function(load_density_np, [dens_path], tf.float32)
    d.set_shape([None, None, 1])
    return img, d

def filter_zero_density(img, dens): return tf.reduce_sum(dens) > 0

def make_dataset(img_dir, dens_dir, batch):
    files = sorted(os.listdir(img_dir))
    ip = [os.path.join(img_dir, f) for f in files]
    dp = [os.path.join(dens_dir, f.replace('.jpg', '.npz')) for f in files]

    ds = tf.data.Dataset.from_tensor_slices((ip, dp))
    ds = ds.map(load_pair_tf, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.filter(filter_zero_density)
    ds = ds.shuffle(1000).batch(batch, drop_remainder=True)
    return ds.prefetch(tf.data.AUTOTUNE)
