import tensorflow as tf
import sys
import time
import glob


def get_tfrecord_dataset(filename):
    raw_dataset = tf.data.TFRecordDataset(filename)
    return raw_dataset


def parse_example(serialized_example):
    feature_context = {
        'image/data': tf.io.FixedLenFeature([], tf.string),
        'image/image_id': tf.io.FixedLenFeature([], tf.int64),
    }
    feature_sequence = {
        'image/caption_ids': tf.io.FixedLenSequenceFeature([], dtype=tf.int64)
    }
    parsed_context, parsed_sequence = tf.io.parse_single_sequence_example(
        serialized=serialized_example,
        context_features=feature_context,
        sequence_features=feature_sequence)

    image = read_image(parsed_context['image/data'])
    image_id = parsed_context['image/image_id']
    caption_id = parsed_sequence['image/caption_ids']

    return image, image_id, caption_id


def read_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(images=image, size=(224, 224))
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1

    return image


def augment(img, idx, cpt):
    transformation = tf.random.uniform(shape=[1],
                                       minval=1,
                                       maxval=4,
                                       dtype=tf.int32)
    if transformation == 1:
        img = tf.image.random_flip_up_down(img)
    elif transformation == 2:
        img = tf.image.random_flip_left_right(img)
    elif transformation == 3:
        img = tf.image.random_hue(img, max_delta=0.2)
    elif transformation == 4:
        img = tf.image.random_brightness(img, max_delta=0.1)

    return img, idx, cpt


def get_dataset(mode=None, batch_size=64):
    if mode == tf.estimator.ModeKeys.TRAIN:
        files = glob.glob('/data/train-*-of-00256')
    elif mode == tf.estimator.ModeKeys.EVAL:
        files = glob.glob('/data/val-*-of-00004')
    elif mode == tf.estimator.ModeKeys.PREDICT:
        files = glob.glob('/data/test-*-of-00008')
    else:
        print('Dataset type not provided\nExiting...')
        sys.exit()

    raw_dataset = get_tfrecord_dataset(files)
    ds = raw_dataset.map(parse_example,
                         num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if mode == tf.estimator.ModeKeys.TRAIN:
        ds_augmented = ds.map(augment,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds.concatenate(ds_augmented)
        ds = ds.shuffle(5000)

    ds = ds.padded_batch(batch_size=batch_size,
                         padded_shapes=([None, None, None], [], [60]),
                         padding_values=(tf.constant(11519, dtype=tf.float32),
                                         tf.constant(11519, dtype=tf.int64),
                                         tf.constant(11519, dtype=tf.int64)),
                         drop_remainder=True)

    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return ds


def main():
    start = time.time()
    dataset = get_dataset(tf.estimator.ModeKeys.TRAIN)
    print(f'Time taken for VAL ds {time.time() - start} sec')
    print(dataset)

    # check if dataset is not empty
    for img, idx, cpt in dataset.take(1):
        print(cpt)


if __name__ == "__main__":
    main()
