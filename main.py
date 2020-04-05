import argparse
import platform
import tensorflow as tf
from create_dataset import get_dataset
from model import Model

# if MAC OS or Windows, do not use GPU
if platform.system() != 'Darwin' or platform.system() != 'Windows':
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU devices available"
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

parser = argparse.ArgumentParser(description="Image Captioning")
parser.add_argument("-m", "--mode",
                    dest="mode",
                    type=str,
                    required=True)
parser.add_argument("-dst", "--dst_ckp",
                    dest="dst_ckp",
                    type=str,
                    default='default_ckpt')
parser.add_argument("-src", "--src_ckp",
                    dest="src_ckp",
                    type=str, default='')
parser.add_argument("-r", "--regularization",
                    dest="regularization",
                    action="store_true",
                    default=False,
                    help="For attention: dropout, else dropout and stacked cells")
parser.add_argument("-uf", "--unfreeze",
                    dest="unfreeze",
                    action="store_true",
                    default=False)
parser.add_argument("-a", "--attention",
                    dest="attention",
                    action="store_true",
                    default=False)
parser.add_argument("-ft", "--fine_tune_at",
                    dest="fine_tune_at",
                    type=int,
                    default=100)
parser.add_argument("-kr", "--kernel_regularizer",
                    dest="kernel_regularizer",
                    action="store_true",
                    default=False)

args = parser.parse_args()

params = {
    'num_epochs': 40,
    'embedding_dims': 512,
    'lstm_units': 512,
    'vocab_size': 11520,
    'batch_size': 64,
    'learning_rate': 0.0005 if args.unfreeze else 2.0,
    'total_samples': 586368,
    'num_epochs_before_decay': 8,
    'decay_rate': 0.5,
    'weight_decay': 0.0005,
    'kernel_regularizer': args.kernel_regularizer,
    'regularization': args.regularization,
    'dropout_rate': 0.2,
    'unfreeze': args.unfreeze,
    'fine_tune_at': args.fine_tune_at,
    'src_ckp': args.src_ckp,
    'dst_ckp': args.dst_ckp,
    'generated_cpts_path': 'results/test_evaluation_result.json',
    'attention': args.attention
}

ds_val = get_dataset(tf.estimator.ModeKeys.EVAL, params['batch_size'])
ds_train = get_dataset(tf.estimator.ModeKeys.TRAIN, params['batch_size'])

model = Model(params)

if args.src_ckp:
    model.restore_checkpoint()
    # print('Scores from restored checkpoint:')
    # model.evaluate(ds_val)
if args.mode.lower() in ['train', 'training']:
    model.training(ds_train, ds_val)
elif args.mode.lower() in ['eval', 'evaluate', 'evaluation']:
    model.evaluate(ds_val)
else:
    print('Invalid mode. Choose from {train, evaluate}')
