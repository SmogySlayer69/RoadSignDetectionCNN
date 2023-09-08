import pathlib
import tensorflow as tf

dataset_url = "need to put images in dataset"
data_dir = tf.keras.utils.get_file(
    origin=dataset_url, fname="", untar=True
)

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir, image_size=(180, 180), batch_size=64
)
