import tensorflow as tf
from core.yolov3 import YOLOV3
from core.utils import load_weights

INPUT_SIZE = 608
TINY = True
darknet_weights = '/home/tk/Desktop/machine_learning/tensorFlowTrain/models_custom/object_detection/tfyolov3/darknet_weights/bird_tiny.weights'
ckpt_file = './checkpoint/bird-tiny.ckpt'

input_data = tf.placeholder(dtype=tf.float32, shape=(None, INPUT_SIZE, INPUT_SIZE, 3), name='inputs')
model = YOLOV3(input_data, trainable=False, tiny=TINY)
load_ops = load_weights(tf.global_variables(), darknet_weights)

saver = tf.train.Saver(tf.global_variables())

with tf.Session() as sess:
    sess.run(load_ops)
    save_path = saver.save(sess, save_path=ckpt_file)
    print('Model saved in path: {}'.format(save_path))
