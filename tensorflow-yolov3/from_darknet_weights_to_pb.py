# -*- coding: utf-8 -*-

import tensorflow as tf
from core.yolov3 import YOLOV3

from core.utils import load_weights, freeze_graph

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    'weights_file',
    '/home/tk/Desktop/machine_learning/tensorFlowTrain/models_custom/object_detection/tfyolov3/darknet_weights/bird_tiny.weights',
    'Binary file with detector weights')
tf.app.flags.DEFINE_string(
    'output_pb', 'pb/bird_tiny_3l.pb', 'Frozen tensorflow protobuf model output path')
tf.app.flags.DEFINE_integer(
    'size', 608, 'Image size')
tf.app.flags.DEFINE_bool(
    'tiny', False, 'Tiny model or not')

def main(argv=None):
    input_data = tf.placeholder(dtype=tf.float32, shape=(None, 608, 608, 3), name='inputs')
    model = YOLOV3(input_data, trainable=False, tiny=FLAGS.tiny)
    load_ops = load_weights(tf.global_variables(), FLAGS.weights_file)

    with tf.Session() as sess:
        sess.run(load_ops)
        freeze_graph(sess, FLAGS.output_graph)


if __name__ == '__main__':
    tf.app.run()
