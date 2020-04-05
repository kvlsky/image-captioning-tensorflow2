import tensorflow as tf
from words import idx2sentence
import json


def write_logs_evaluation(prediction, caption, image, step):
    sentence = [' '.join(idx2sentence(p)) for p in prediction]
    ground_truth = caption.numpy().tolist()
    ground_truth = [' '.join(idx2sentence(g)) for g in ground_truth]
    for i in range(5):
        tf.summary.image(name=ground_truth[i],
                         data=[image[i]],
                         step=step,
                         description=sentence[i],
                         max_outputs=1)


def write_logs_training(loss, learning_rate, scores, epoch):
    tf.summary.scalar('loss', loss, step=epoch)
    tf.summary.scalar('learning_rate', learning_rate, step=epoch)
    for k, v in scores.items():
        tf.summary.scalar(k, v, step=epoch)


def save_json(path, data):
    with open(path, 'w+') as json_file:
        json.dump(data, json_file)
