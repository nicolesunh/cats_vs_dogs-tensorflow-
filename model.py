import tensorflow as tf


# 一个简单的卷积神经网络，卷积+池化层x2，全连接层x2，最后一个softmax层做分类。
def inference(images, batch_size, n_classes):
    '''Bulid the model
    Args:
        n_classes = 2：二分类
        images: image batch, 4D tensor, tf.float32, [batch_size, width, height, channels]
    Returns:
        output tensor with the computed logits, float, [batch_size, n_classes]
    '''
    # cov1, shape = [kernel size, kernel size, channels, kernel numbers]
    with tf.variable_scope('conv1') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3, 3, 3, 16],  # 16[kernel numbers]：跟计算的精度有关 16个3x3的卷积核（3通道）
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,
                                                                              dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))  # 初始化0.1
        conv = tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)

    # pool1 and norm1 3x3最大池化，步长strides为2，池化后执行lrn()操作，局部响应归一化，对训练有利
    with tf.variable_scope('poling_lrn') as scope:
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],  # 标准数值
                               padding='SAME', name='pooling1')
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0,  # 局部响应归一化函数。标准数值
                          beta=0.75, name='norm1')

    # conv2 
    with tf.variable_scope('conv2') as scope:
        weights = tf.get_variable('weights', shape=[3, 3, 16, 16], #16个3x3的卷积核（16通道）
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,
                                                                              dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(norm1, weights, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name='conv2')

    # pool2 and norm2
    with tf.variable_scope('pooling2_lrn') as scope:
        norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0,
                          beta=0.75, name='norm2')
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1],
                               padding='SAME', name='pooling2')

    # local3 128个神经元，将之前pool层的输出reshape成一行，激活函数relu()
    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(pool2, shape=[batch_size, -1])  # 变换成向量
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable('weights',
                                  shape=[dim, 128],  # 全连接个数，128
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005,
                                                                              dtype=tf.float32))

        biases = tf.get_variable('biases',
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

    # local4
    with tf.variable_scope('local4') as scope:
        weights = tf.get_variable('weights',
                                  shape=[128, 128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005,
                                                                              dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name='local4')

    # softmax 将前面的FC层输出，做一个线性回归，计算出每一类的得分，在这里是2类，所以这个层输出的是两个得分。
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.get_variable('softmax_linear',
                                  shape=[128, n_classes],  # n_classes:2，表示二分类
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005,
                                                                              dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[n_classes],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name='softmax_linear')

    return softmax_linear


# %%
def losses(logits, labels):
    with tf.variable_scope('lose') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits \
            (logits=logits, labels=labels, name='xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name + '/loss', loss)

    return loss


# %% 训练优化
def training(loss, learning_rate):
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op


# %%
def evalution(logits, labels):
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)  # 取最大值
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name + '/accuracy', accuracy)
    return accuracy