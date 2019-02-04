#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from mask_rcnn.client import Client
import numpy as np
import cv2
from cv_bridge import CvBridge

rospy.init_node('visualize_mask_rcnn')
mask_rcnn = Client()
bridge = CvBridge()

def cb(imgmsg):
    #vis = mask_rcnn.visualize_imgmsg(imgmsg)
    image = bridge.imgmsg_to_cv2(imgmsg, 'bgr8')
    masks = mask_rcnn.process_image(image, 'bgr8')
    total = np.zeros(image.shape[:2]+(1,), dtype=np.float32)
    for mask in masks:
        if mask.name != 'person':
            continue
        print '%s (%f%%)' % (mask.name, mask.probability*100)
        total += np.expand_dims(mask.mask, -1) * mask.probability
    total[total>1.] = 1.
    #total[total<1.5] = 0.
    cv2.imshow('MaskRCNN', np.uint8(image*total))
    cv2.imshow('original', image)
    cv2.waitKey(1)

rospy.Subscriber('image', Image, cb)
rospy.spin()
