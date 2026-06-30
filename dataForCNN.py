import tensorflow as tf
import pandas

def load_dataset(dataframe,batch_size,is_training=True,augment=False):
    paths = dataframe['path'].values
    labels = dataframe['label'].values
    dataset = tf.data.Dataset.from_tensor_slices((paths,labels))
    def parse(file_path,label):
        image = tf.io.read_file(file_path)
        image = tf.image.decode_jpeg(image,channels=3)
        image = tf.image.resize(image,[224,224])
        if augment:
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_flip_up_down(image)
            image = tf.image.rot90(image,tf.random.uniform(shape=[],minval=0,maxval=4,dtype=tf.int32))
            image = tf.image.random_brightness(image,0.1)
            image = tf.image.random_contrast(image,0.8,1.2)
            image = tf.image.random_saturation(image,0.8,1.2)
        image = tf.cast(image,tf.float32)/255.0
        return image,label
    dataset = dataset.map(parse,num_parallel_calls=1)
    if is_training:
        dataset = dataset.shuffle(buffer_size=min(1000,len(dataframe)))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)
    return dataset
