import os
import numpy as np
import tensorflow as tf
import  input_data
import model


N_CLASSES = 2
IMG_W = 208
IMG_H = 208
BATCH_SIZE = 16
CAPACITY = 2000  # 队列中元素个数
MAX_STEP = 5000
learning_rate = 0.0001  # 小于0.001

#获取批次batch
train_dir = 'D:/PychamProjects/Cats_Dogs/data/train/'  # 训练图片文件夹
logs_train_dir = 'D:/PychamProjects/Cats_Dogs/logs/train'  # 保存训练结果文件夹

train, train_label = input_data.get_files(train_dir)

train_batch, train_label_batch = input_data.get_batch(train,
                                                       train_label,
                                                       IMG_W,
                                                       IMG_H,
                                                       BATCH_SIZE,
                                                       CAPACITY)
#操作定义
train_logits = model.inference(train_batch, BATCH_SIZE, N_CLASSES)
train_loss = model.losses(train_logits, train_label_batch)
train_op = model.training(train_loss, learning_rate)
train_acc = model.evalution(train_logits, train_label_batch)

summary_op = tf.summary.merge_all()
sess = tf.Session()
#writer:写Log文件
train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
#产生一个saver来存储训练好的模型
saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())
#队列监控
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)



# 开始训练
try:
    # 执行MAX_STEP步的训练，一步一个batch
    for step in np.arange(MAX_STEP):
        if coord.should_stop():
            break
        _, tra_loss, tra_acc = sess.run([train_op, train_loss, train_acc])

        if step % 50 == 0:
            print('Step %d, train loss = %.2f, train accuracy = %.2f%%' % (step, tra_loss, tra_acc))
            summary_str = sess.run(summary_op)
            train_writer.add_summary(summary_str, step)

        if step % 2000 == 0 or (step + 1) == MAX_STEP:
            checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)

except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')
finally:
    coord.request_stop()

coord.join(threads)
sess.close()