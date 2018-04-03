import tensorflow as tf
import numpy as np


def tf_slice(x, selector):
    assert len(selector) == 2
    if not type(selector[1]) is slice and not type(selector[1]) is int:
        x = tf.transpose(x, [0, 2, 1])
        y = x[(selector[1], selector[0])]
        return tf.transpose(y, [0, 2, 1])
    return x[selector]


def dilated_res(x, cond_x, nb_filters, kernel_size, input_length, nb_skips, dilation, scope="dilated"):
    h = x
    with tf.variable_scope(scope):
        data_out = tf.layers.conv1d(h,
                                    filters=2 * nb_filters,
                                    kernel_size=kernel_size,
                                    padding="SAME",
                                    dilation_rate=[dilation],
                                    use_bias=False)

        acts = [tf.nn.tanh, tf.nn.sigmoid]
        sliced_data = [tf_slice(data_out, (..., slice(0, nb_filters))),
                       tf_slice(data_out, (..., slice(nb_filters, 2 * nb_filters)))]
        cond_x = tf.layers.dense(cond_x, units=2 * nb_filters, use_bias=False)
        cond_x = tf.reshape(cond_x, shape=[-1, nb_filters, 2])
        sliced_cond = [cond_x[..., 0], cond_x[..., 1]]
        outs = []
        for cond, data, act in zip(sliced_cond, sliced_data, acts):
            cond_out = tile2d(cond, input_length)
            data_out = tf.add(data, cond_out)
            outs.append(act(data_out))

        data_x = tf.multiply(*outs)

        data_x = tf.layers.conv1d(data_x, nb_filters + nb_skips, kernel_size=1, padding="SAME", use_bias=False)
        res_x = tf_slice(data_x, (..., slice(0, nb_filters)))
        skip_x = tf_slice(data_x, (..., slice(nb_filters, nb_filters + nb_skips)))
        res_x = tf.add(x, res_x)
    return res_x, skip_x


def tile2d(x, nb_repeats):
    assert x.shape.ndims == 2
    return tf.tile(tf.expand_dims(x, 1), multiples=tf.stack([1, nb_repeats, 1]))


def block(x, other, nb_filters, kernel_size, nb_repeats,
          use_bias=False,
          init=tf.initializers.orthogonal,
          scope="block"):
    # conv1d
    # fc
    # tile
    # add
    with tf.variable_scope(scope):
        x = tf.layers.conv1d(x, filters=nb_filters, kernel_size=kernel_size, padding="SAME", use_bias=use_bias,
                             kernel_initializer=init)
        other = tf.layers.dense(other, units=nb_filters, use_bias=use_bias, kernel_initializer=init)
        other = tile2d(other, nb_repeats)
    return tf.add(x, other)


def add_singleton_depth(x):
    x = tf.expand_dims(x, axis=-1)
    if x.shape.ndims == 4:
        return tf.transpose(x, [0, 3, 1, 2])
    else:
        return x


def get_cond_x_dim(representation, num_condition_classes):
    if representation == 'binary':
        return int(np.max((np.ceil(np.log2(num_condition_classes)), 1)))
    else:
        return num_condition_classes


def get_idx(input_length, target_field_length, target_padding=0):
    target_sample_index = int(np.floor(input_length / 2.0))

    return range(target_sample_index - (target_field_length // 2) - target_padding,
                 target_sample_index + (target_field_length // 2) + target_padding + 1)


def get_r_field_dim(stacks, dilations, filter_length, target_field_length=1):
    half_filter_length = (filter_length - 1) // 2
    length = 0
    for d in dilations:
        length += d * half_filter_length
    length = 2 * length
    length = stacks * length
    length += target_field_length
    return length



class Model(object):
    def __init__(self, config, n_classes, scope="main"):
        self.config = config
        self.scope = scope
        self.dilations = [2 ** i for i in range(self.config.dilations + 1)]

        self.obs_dim_cond = get_cond_x_dim(self.config.encoding, n_classes)
        r_field_dim = get_r_field_dim(self.config.num_stacks,
                                      self.dilations,
                                      self.config.lengths.res)
        self.obs_dim = r_field_dim + (self.config.target_field_length - 1)
        self.x_sample_idx = get_idx(self.obs_dim, self.config.target_field_length, self.config.target_padding)
        self.y_sample_idx = get_idx(self.obs_dim, self.config.target_field_length)
        #
        self.y_dim_w_pad = self.config.target_field_length + 2 * self.config.target_padding
        self._init_graph()

    def _init_graph(self):
        self._gs = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")
        self._init_ph()
        self._build_graph()
        self._loss_op()
        self._train_op()
        self._fetches = [self.l1, self.l2]
        self._summary_op()

    def _init_ph(self):

        self.x = tf.placeholder(tf.float32, shape=[None, self.obs_dim], name="x")
        self.x_cond = tf.placeholder(tf.float32, shape=[None, self.obs_dim_cond], name="x_cond")
        self.y = tf.placeholder(tf.float32, shape=[None, self.y_dim_w_pad], name="y")
        self.y_noise = tf.placeholder(tf.float32, shape=[None, self.y_dim_w_pad], name="y_speech")

    def _build_graph(self, act=tf.nn.relu):

        h = add_singleton_depth(self.x)
        h_cond = self.x_cond
        h_target_field = tf_slice(h, (slice(self.x_sample_idx[0], self.x_sample_idx[-1] + 1, 1), ...))

        # block 1
        with tf.variable_scope(self.scope):
            h = block(h, h_cond,
                      self.config.depths.res,
                      self.config.lengths.res,
                      self.obs_dim, scope="block_1")

            skip_connections = []
            for stack in range(self.config.num_stacks):
                for dilation in self.dilations:
                    _, skip_out = dilated_res(h, h_cond,
                                              self.config.depths.res,
                                              self.config.lengths.res,
                                              self.obs_dim,
                                              self.config.depths.skip,
                                              dilation, scope="s_{}_w_d{}".format(stack, dilation))

                    if skip_out is not None:
                        skip_out = tf_slice(skip_out, (slice(self.x_sample_idx[0], self.x_sample_idx[-1] + 1, 1), ...))
                        skip_connections.append(skip_out)
            # skip connections
            h = act(tf.add_n(skip_connections))
            # block 2
            h = act(block(h, h_cond,
                          self.config.depths.final[0],
                          self.config.lengths.final[0],
                          self.y_dim_w_pad, scope="block_2"))
            # block 3
            h = block(h, h_cond,
                      self.config.depths.final[1],
                      self.config.lengths.final[1],
                      self.y_dim_w_pad, scope="block_3")
            # out
            h = tf.layers.conv1d(h, filters=1, kernel_size=1)
            h_speech = h
            h_noise = tf.subtract(h_target_field, h_speech)
            self.y_hat = tf.squeeze(h_speech, 2)
            self.y_hat_noise = tf.squeeze(h_noise, 2)

    def _loss_op(self):
        self.l1 = tf.reduce_mean(tf.abs(self.y - self.y_hat))
        self.l2 = tf.reduce_mean(tf.abs(self.y_noise - self.y_hat_noise))
        self.loss = self.l1 + self.l2
        self.vmae = tf.reduce_mean(tf.abs(self.y[:, 1:-2] - self.y_hat[:, 1:-2]))

    def _train_op(self):
        self._optim = tf.train.AdamOptimizer(learning_rate=self.config.lr)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # Ensures that we execute the update_ops before performing the train_step
            self.train = self._optim.minimize(self.loss,
                                              var_list=tf.trainable_variables(self.scope),
                                              global_step=self._gs, name="step")

    def _summary_op(self):
        ops = []
        for t in self._fetches:
            if t.get_shape().ndims >= 1:
                ops.append(tf.summary.histogram(name=t.name, values=t))
            elif t.get_shape().ndims < 1:
                ops.append(tf.summary.scalar(name=t.name, tensor=t))
            else:
                print("Shape not found")
        self.summarize = tf.summary.merge_all()


if __name__ == "__main__":
    config = Config(model)
    model = Model(config, 29)
