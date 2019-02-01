#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from mask_rcnn.msg import MaskArray, Mask
from mask_rcnn.srv import GetNames, GetNamesResponse
from mask_rcnn.srv import DetectObjects, DetectObjectsResponse
from cv_bridge import CvBridge

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

import mask_rcnn.configs.config_v1 as cfg
import mask_rcnn.datasets.dataset_factory as datasets
import mask_rcnn.nets.nets_factory as network
import mask_rcnn.nets.pyramid_network as pyramid_network

import cv2

FLAGS = tf.app.flags.FLAGS

# To be configurable in future update
CLASS_NAME = ['background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
              'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
              'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
              'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
              'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
              'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
              'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
              'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
              'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
              'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
              'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
              'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
              'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
              'scissors', 'teddy bear', 'hair drier', 'toothbrush']

class FastMaskRCNN:
    def __init__(self, ckpt_path):
        self.make_graph()
        self.make_session(ckpt_path)
        self.bridge = CvBridge()
        self.mask_pub = rospy.Publisher('masks', MaskArray, queue_size=1)
        self.image_sub = rospy.Subscriber('image', Image, self.image_callback)
        rospy.Service('get_names', GetNames, self.get_names)
        rospy.Service('detect_objects', DetectObjects, self.detect_objects)

    def image_callback(self, imgmsg):
        out = self.process_imgmsg(imgmsg)
        self.mask_pub.publish(out)

    def get_names(self, req):
        return GetNamesResponse(names=CLASS_NAME)

    def detect_objects(self, req):
        out = self.process_imgmsg(req.image, req.nbest)
        return DetectObjectsResponse(masks=out)

    # TODO: support sorting when nbest>=0
    def process_imgmsg(self, imgmsg, nbest=-1):
        image = self.bridge.imgmsg_to_cv2(imgmsg, 'bgr8')
        #h, w = orig_image.shape[:2]
        #new_h, new_w = _smallest_size_at_least(h, w, FLAGS.image_min_size)
        #image = cv2.resize(orig_image, (new_w,new_h))
        #cv2.resize(orig_image, (640,640))
        image = np.float32(image) / 256.
        image = image * 2. - 1.
        image = np.expand_dims(image, 0)

        fetch_list = [self.box, self.mask, self.cls, self.score]
        feed_dict = {self.image_ph: image}
        outputs = self.sess.run(fetch_list, feed_dict)

        masks = MaskArray()
        masks.header.stamp = imgmsg.header.stamp
        masks.header.frame_id = imgmsg.header.frame_id
        #shown = False
        for box, mask, cls, score in zip(*outputs):
            if cls == 0:
                continue
            x1, y1, x2, y2 = box
            maskmsg = Mask()
            left = min(x1, x2)
            top = min(y1, y2)
            right = max(x1, x2)
            bottom = max(y1, y2)
            maskmsg.left = left
            maskmsg.right = right
            maskmsg.top = top
            maskmsg.bottom = bottom
            maskmsg.probability = score
            maskmsg.class_id = cls
            mask_u8 = np.uint8(mask[:,:,cls]*255)
            maskmsg.image = self.bridge.cv2_to_imgmsg(mask_u8, 'mono8')
            masks.masks.append(maskmsg)
            #if not shown:
            #    x1, y1, x2, y2 = map(int, box)
            #    msk = np.expand_dims(mask_u8, 2)/255.
            #    img = np.uint8((image[0]+1)*128*msk)
            #    cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0))
            #    cv2.imshow('Masked image', img)
            #    shown = True
            #    cv2.waitKey(1)
        return masks

    def make_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.image_ph = tf.placeholder(tf.float32, shape=[None,None,None,3])
            im_shape = tf.shape(self.image_ph)
            _, end_points, pyramid_map = network.get_network(FLAGS.network, self.image_ph, is_training=True) # is_training is set as a workaround. Maybe batch_norm problem.
            pyramid = pyramid_network.build_pyramid(pyramid_map, end_points)
            outputs = pyramid_network.build_heads(pyramid,
                                                  im_shape[1], im_shape[2],
                                                  num_classes=81,
                                                  base_anchors=9,
                                                  is_training=False)
            self.box = outputs['final_boxes']['box']
            self.prob = outputs['final_boxes']['prob']
            self.mask = tf.nn.sigmoid(outputs['mask']['mask'])
            self.cls = outputs['mask']['cls']
            self.score = outputs['mask']['score']
            self.saver = tf.train.Saver()
        self.graph.finalize()

    def make_session(self, ckpt_path):
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(graph=self.graph,
                               config=tf.ConfigProto(gpu_options=gpu_options))
        self.saver.restore(self.sess, ckpt_path)

def main():
    import os, sys, rospkg
    pkg = rospkg.rospack.RosPack().get_path('mask_rcnn')
    ckpt_path = os.path.join(pkg, 'data', 'coco_resnet50_model.ckpt-2499999')
    #if len(sys.argv) > 1:
    #    ckpt_path = sys.argv[1]

    rospy.init_node('mask_rcnn')
    FastMaskRCNN(ckpt_path)
    rospy.spin()

if __name__ == '__main__':
    main()
