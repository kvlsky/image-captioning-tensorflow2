import tensorflow as tf
import datetime
import time
import os
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, LSTMCell, Embedding, Dropout, LSTM
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from words import get_index, START_WORD, STOP_WORD, format_mscoco
from mscoco_eval import eval_scores
from log import write_logs_evaluation, write_logs_training, save_json

start_word_index = get_index(START_WORD)
stop_word_index = get_index(STOP_WORD)

BATCH_SIZE = 64


class CNN_Encoder(tf.keras.Model):
    def __init__(self, params):
        super(CNN_Encoder, self).__init__()
        self.params = params
        self.base_model = MobileNetV2(input_shape=(224, 224, 3),
                                      weights='imagenet',
                                      include_top=False)
        if self.params['unfreeze']:
            self.base_model.trainable = True
            for layer in self.base_model.layers[:self.params['fine_tune_at']]:  # total of 155 layers
                layer.trainable = False
            print(f'Unfreezed CNN-Encoder Layers at Layer: {self.params["fine_tune_at"]}')

        else:
            self.base_model.trainable = False
            print('Freezed CNN-Encoder')
        self.pool = GlobalAveragePooling2D()
        self.fc = Dense(self.params['embedding_dims'], activation='relu')

    def call(self, x):
        x = self.base_model(x, training=self.params['unfreeze'])
        x = self.pool(x)
        x = self.fc(x)
        return x


class RNN_Decoder(tf.keras.Model):
    def __init__(self, params):
        super(RNN_Decoder, self).__init__()
        self.params = params
        self.embedding = Embedding(params['vocab_size'],
                                   params['embedding_dims'])
        k_reg = tf.keras.regularizers.l2(params['weight_decay']) if self.params['kernel_regularizer'] else None
        if params['regularization']:
            print('stacked cell')
            lstm_cell1 = tf.keras.layers.LSTMCell(units=params['lstm_units'], kernel_regularizer=k_reg)
            lstm_cell2 = tf.keras.layers.LSTMCell(units=params['lstm_units'], kernel_regularizer=k_reg)
            lstm_cell3 = tf.keras.layers.LSTMCell(units=params['lstm_units'], kernel_regularizer=k_reg)

            self.lstm = tf.keras.layers.StackedRNNCells([lstm_cell1, lstm_cell2, lstm_cell3])
        else:
            self.lstm = LSTMCell(params['lstm_units'], kernel_regularizer=k_reg)
        self.fc = Dense(params['vocab_size'], kernel_regularizer=k_reg)

        self.dropout = Dropout(params['dropout_rate'])

    def call(self, dec_input, hidden):
        embedded_caption = self.embedding(dec_input)
        # [batch_size, 1, embedding_dims] -> [batch_size, embedding_dims]
        embedded_caption = tf.reshape(embedded_caption,
                                      [dec_input.shape[0], -1])

        x, hidden = self.lstm(embedded_caption, hidden)
        if self.params['regularization']:
            x = self.dropout(x)
        x = self.fc(x)

        return x, hidden

    def initialize_lstm(self, features):
        zero_state = self.lstm.get_initial_state(batch_size=features.shape[0],
                                                 dtype=tf.float32)
        _, hidden = self.lstm(features, zero_state)
        return hidden

# Idea taken from:
# https://www.tensorflow.org/tutorials/text/image_captioning
class Attention(tf.keras.Model):
    def __init__(self, units):
        super(Attention, self).__init__()
        self.layer1 = Dense(units)
        self.layer2 = Dense(units)
        self.v = Dense(1)

    def call(self, features, hidden):
        h_w_t_a = tf.expand_dims(hidden, 1)
        score = tf.nn.tanh(self.layer1(features) + self.layer2(h_w_t_a))
        attention_weights = tf.nn.softmax(self.v(score), axis=1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights


class Attention_Decoder(RNN_Decoder):
    def __init__(self, params):
        super(Attention_Decoder, self).__init__(params)
        self.attention = Attention(self.params['lstm_units'])
        self.features = None
        
        # We don't want the LSTMCell from RNN_Decoder here:
        self.lstm = LSTM(self.params['lstm_units'],
                         return_sequences=True,
                         return_state=True)

    def call(self, dec_input, hidden):
        context_vector, attention_weights = self.attention(self.features, hidden)
        x = self.embedding(dec_input)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        x, state, c = self.lstm(x)
        if self.params['regularization']:
            x = self.dropout(x)
        x = self.fc(x)
        return x, state

    def initialize_lstm(self, features):
        self.features = features
        zero_state = tf.zeros((BATCH_SIZE, self.params['lstm_units']))
        return zero_state


class Model:
    def __init__(self, params):
        self.params = params
        self.encoder = CNN_Encoder(self.params)
        if params['attention']:
            self.decoder = Attention_Decoder(self.params)
        else:
            self.decoder = RNN_Decoder(self.params)

        self.loss_object = SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

        if params['unfreeze']:
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=params['learning_rate'])
        elif params['attention']:
            self.optimizer = tf.keras.optimizers.Adam()
        else:
            steps_to_decay = (params['total_samples'] / params['batch_size']) \
                             * params['num_epochs_before_decay']
            lr_decay_fn = ExponentialDecay(initial_learning_rate=params['learning_rate'],
                                           decay_steps=steps_to_decay,
                                           decay_rate=params['decay_rate'],
                                           staircase=True)
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=lr_decay_fn)

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')

        # Checkpoint and Tensorboard
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = './tf_logs/' + current_time + '_train'
        self.summary_writer = tf.summary.create_file_writer(train_log_dir)

        self.ckpt = tf.train.Checkpoint(encoder=self.encoder,
                                        decoder=self.decoder,
                                        optimizer=self.optimizer)
        self.dst_manager = tf.train.CheckpointManager(self.ckpt,
                                                      params['dst_ckp'],
                                                      max_to_keep=2)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[BATCH_SIZE, 224, 224, 3],
                      dtype=tf.float32),
        tf.TensorSpec(shape=[BATCH_SIZE, 60], dtype=tf.int64)])
    def train_step(self, img, caption):
        loss = 0

        with tf.GradientTape() as tape:
            features = self.encoder(img)
            hidden = self.decoder.initialize_lstm(features)

            for i in range(caption.shape[1] - 1):
                dec_input = tf.expand_dims(caption[:, i], 1)
                pred, hidden = self.decoder(dec_input, hidden)

                loss += self.loss_function(caption[:, i + 1], pred)

        trainable_variables = self.encoder.trainable_variables + \
                              self.decoder.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)

        if self.params['unfreeze']:
            self.optimizer.apply_gradients(zip(gradients,
                                               trainable_variables))
        else:
            clipped_gradients, global_norm = tf.clip_by_global_norm(gradients, 5.0)
            self.optimizer.apply_gradients(zip(clipped_gradients,
                                               trainable_variables))

        self.train_loss(loss)

        return loss / int(caption.shape[1])

    @tf.function(input_signature=[tf.TensorSpec(shape=[BATCH_SIZE, 224, 224, 3], dtype=tf.float32)])
    def inference_step(self, img):
        features = self.encoder(img)

        hidden = self.decoder.initialize_lstm(features)
        input_ = tf.tile(tf.constant([start_word_index], dtype=tf.int64), [self.params['batch_size']])
        predictions = tf.identity(input_)
        predictions = tf.reshape(predictions, [1, -1])
        input_ = tf.reshape(input_, [-1, 1])

        for _ in range(60):
            output, hidden = self.decoder(input_, hidden)
            pred = self.get_prediction(output)
            input_ = pred
            pred = tf.reshape(pred, [1, -1])
            predictions = tf.concat([predictions, pred], 0)

        return predictions

    def get_prediction(self, logits):
        return tf.argmax(tf.nn.softmax(logits), axis=-1)

    def kernel_reg_loss(self):
        if not self.params['kernel_regularizer']:
            return tf.constant(0, dtype=tf.float32)
        else:  # L2 Regularization
            return tf.math.add_n(self.decoder.losses)  # Model Extension part b), our own idea.

    def loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 11519))
        loss = self.loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask

        return tf.math.add_n([tf.reduce_mean(loss), self.kernel_reg_loss()])

    def evaluate(self, ds):
        json_data = []
        unique_ids = []

        for step, sample in enumerate(ds):
            img, idx, cpt = sample
            predictions = self.inference_step(img)
            #predictions = tf.transpose(predictions).numpy()
            predictions = tf.transpose(predictions).numpy().tolist()
            for i, p in enumerate(predictions):
                if stop_word_index in predictions[i]:
                    predictions[i] = p[:p.index(stop_word_index)]

            for i, p in enumerate(predictions):
                if idx[i] in unique_ids:
                    continue
                caption = format_mscoco(p)
                pred_dict = {'image_id': int(idx.numpy()[i]), 'caption': caption}
                json_data.append(pred_dict)

                unique_ids.append(idx[i])

            if step == 0:
                with self.summary_writer.as_default():
                    write_logs_evaluation(predictions, cpt, img, step)

        if not os.path.exists('results'):
            os.makedirs('results')
        save_json(path=self.params['generated_cpts_path'], data=json_data)

        scores = eval_scores(res_file='results/test_evaluation_result.json')

        return scores

    def restore_checkpoint(self):
        manager = tf.train.CheckpointManager(self.ckpt, self.params['src_ckp'], max_to_keep=5)
        if manager.latest_checkpoint:
            status = self.ckpt.restore(manager.latest_checkpoint)
            status.assert_existing_objects_matched()
            status.expect_partial()
            print(f'Checkpoint restored from: {manager.latest_checkpoint}')
        else:
            print('Did not load checkpoints.')

    def training(self, ds_train, ds_val):
        for epoch in range(1,self.params['num_epochs']+1):
            print(f'Starting epoch {epoch}.')
            start = time.time()
            for batch in ds_train:
                img, _, caption = batch
                self.train_step(img, caption)

            print(f'Train epoch: {epoch}/{self.params["num_epochs"]} Loss: {self.train_loss.result()}')
            print(f'Time taken for epoch {epoch}: {time.time() - start} sec.')
            # evaluate
            if epoch % 5 == 0 or epoch == self.params['num_epochs']:
                print('Calculating scores from generated captions.')
                scores_dict = self.evaluate(ds_val)
                with self.summary_writer.as_default():
                    if self.params['unfreeze']:
                        write_logs_training(loss=self.train_loss.result(),
                                            learning_rate=self.params['learning_rate'],
                                            scores=scores_dict,
                                            epoch=epoch)
                    else:
                        write_logs_training(loss=self.train_loss.result(),
                                            learning_rate=self.optimizer._decayed_lr(tf.float32),
                                            scores=scores_dict,
                                            epoch=epoch)

            self.train_loss.reset_states()
            self.dst_manager.save(checkpoint_number=epoch)
