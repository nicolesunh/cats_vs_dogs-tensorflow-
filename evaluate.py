

import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import model
import input_data



 
def get_one_image(train):
    n = len(train)
    ind = np.random.randint(0,n)
    img_dir = train[ind]
    
    image = Image.open(img_dir)
    plt.imshow(image)
    plt.show()
    image = image.resize([208, 208])
    image = np.array(image)
    
    return image
 
    
train_dir = 'D:/PychamProjects/Cats_Dogs/data/train/'
train, train_label = input_data.get_files(train_dir)
image_array = get_one_image(train)
 
with tf.Graph().as_default():
    BATCH_SIZE = 1
    N_CLASSES = 2
    
    image = tf.cast(image_array, tf.float32) # 转换图片格式
    image = tf.reshape(image, [1, 208, 208, 3]) # 修改图片大小
    logit = model.inference(image, BATCH_SIZE, N_CLASSES) #
    logit = tf.nn.softmax(logit) #激活函数
    
    x = tf.placeholder(tf.float32, shape = [208, 208, 3]) # 
    
    logs_train_dir = 'D:/PychamProjects/Cats_Dogs/logs/train/'
    
    saver = tf.train.Saver()
    
    # 下载训练好的模型        
    with tf.Session() as sess:
        # 下载模型。。。。。。。
        print("Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(logs_train_dir)
        if ckpt and ckpt.model_checkpoint_path:
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('.')[-1]
            saver.restore(sess, ckpt.model_checkpoint_path) # 重载模型
            print('Loading success, global_step is %s' % global_step)
        else:
            print("No checkpoint file found")
            
        prediction = sess.run(logit, feed_dict = {x: image_array})
        max_index = np.argmax(prediction) # 得到prediction的索引
        if max_index == 0:
            print('This is a cat with possiblity %.6f' %prediction[:, 0])
        else:
            print('This is a dog with possiblity %.6f' %prediction[:, 1])



