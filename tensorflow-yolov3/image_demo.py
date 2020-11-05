#! /usr/bin/env python
# coding=utf-8
import cv2
import numpy as np
import core.utils as utils
import tensorflow as tf
import time

return_elements = ["inputs:0", "output_boxes:0"]
pb_file = "./pb/testtiny.pb"
image_path = "./data/dataset/test/w20190829173008684_322.jpg"
classfile = "./data/classes/bird.names"
num_classes = 2
input_size = 608
classes = utils.load_names(classfile)
graph = tf.Graph()

original_image = cv2.imread(image_path)
original_image_size = original_image.shape[:2]
image_data, _ = utils.image_preporcess(np.copy(original_image), [input_size, input_size], keep_aspect_ratio=False)
image_data = image_data[np.newaxis, ...]

return_tensors = utils.read_pb_return_tensors(graph, pb_file, return_elements)

t0 = time.time()
with tf.Session(graph=graph) as sess:
    detected_boxes = sess.run(return_tensors[1], feed_dict={return_tensors[0]: image_data})
print("Predictions found in {:.2f}s".format(time.time() - t0))

filtered_boxes = utils.non_max_suppression(detected_boxes,
                                           confidence_threshold=0.35,
                                           iou_threshold=0.3)[0]

utils.draw_boxes_cv2(filtered_boxes, original_image, classes, (input_size, input_size), keep_aspect_ratio=False)
print('\n\n\n')
cv2.imshow('frame', original_image)
if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
