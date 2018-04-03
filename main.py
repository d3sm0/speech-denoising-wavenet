from config import *

from datasets import Dataset
from model import Model

import tensorflow as tf

dataset_config = Config(dataset_config)
model_config = Config(model_config)
train_config = Config(train_config)

model = Model(model_config, dataset_config.num_classes)
dataset = Dataset(dataset_config, input_length=model.obs_dim)

writer = tf.summary.FileWriter(train_config.log_path)
saver = tf.train.Saver(var_list=tf.trainable_variables(model.scope))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    losses = []
    ep = 0
    t = 0
    while ep < train_config.num_epochs:
        while t < train_config.num_train_samples:
            batch = dataset.sample_batch(buffer_size=train_config.batch_size,
                                         y_size=model.y_dim_w_pad)
            feed_dict = {
                model.x: batch["x"],
                model.x_cond: batch["x_cond"],
                model.y: batch["y"],
                model.y_noise: batch["y_noise"]
            }
            loss, _ = sess.run([model.loss, model.train], feed_dict)
            losses.append(loss)
            t += 1

            if t % train_config.summarize_every == 0:
                # summary, gs = sess.run([model.summarize, model._gs], feed_dict)
                # writer.add_summary(summary, global_step=gs)
                # writer.flush()

                print(ep, t, loss, sum(losses) / t, sep="\t")
        ep += 1
        t = 0
        losses = []
        if ep % train_config.save_every == 0:
            saver.save(sess, save_path=train_config.save_path +"/model.ckpt", write_meta_graph=False)
