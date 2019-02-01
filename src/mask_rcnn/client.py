import rospy
from mask_rcnn.srv import GetNames, DetectObjects
import cv2
import numpy as np
from cv_bridge import CvBridge

class Mask:
    def __init__(self, bbox, name, prob, mask):
        self.left, self.top, self.right, self.bottom = bbox
        self.mask = mask
        self.name = name
        self.probability = prob

    def draw(self, image, color=(0,0,255), alpha=0.5):
        if len(image.shape) == 2:
            image = np.expand_dims(image, -1)
        mask = np.expand_dims(self.mask, -1) * alpha
        color = np.reshape(color, (1,1,3))
        image =  image * (1.-mask) + color * mask
        return np.uint8(image)
        
class Client:
    def __init__(self, get_names_srv='get_names',
                 detect_objects_srv='detect_objects'):
        self.bridge = CvBridge()
        get_names = rospy.ServiceProxy(get_names_srv, GetNames)
        get_names.wait_for_service()
        self.class_names = get_names().names
        self.detect_objects = rospy.ServiceProxy(detect_objects_srv, DetectObjects)
        self.detect_objects.wait_for_service()

    def visualize_image(self, image, encoding='rgb8'):
        imgmsg = self.bridge.cv2_to_imgmsg(image, encoding)
        for mask in self.process_imgmsg(imgmsg):
            image = mask.draw(image)
        return image
    
    def visualize_imgmsg(self, imgmsg):
        image = self.bridge.imgmsg_to_cv2(imgmsg, 'bgr8')
        for mask in self.process_imgmsg(imgmsg):
            image = mask.draw(image)
        return image

    def process_image(self, image, encoding='rgb8'):
        imgmsg = self.bridge.cv2_to_imgmsg(image, encoding)
        return self.process_imgmsg(imgmsg)

    def process_imgmsg(self, imgmsg):
        maskmsg = self.detect_objects(imgmsg, -1).masks
        return self.process_maskmsg(maskmsg, (imgmsg.width, imgmsg.height))
        
    def process_maskmsg(self, maskmsg, imgsize):
        width, height = imgsize
        masks = []
        
        for mask in maskmsg.masks:
            maskimg = self.bridge.imgmsg_to_cv2(mask.image, 'mono8')
            sx = (mask.right-mask.left)/maskimg.shape[1]
            sy = (mask.bottom-mask.top)/maskimg.shape[0]
            M = np.float32([[sx, 0,  mask.left],
                            [0,  sy, mask.top]])
            maskimg = cv2.warpAffine(maskimg, M, imgsize)/255.
            bbox = (mask.left, mask.top, mask.right, mask.bottom)
            masks.append(Mask(bbox,
                              self.class_names[mask.class_id],
                              mask.probability,
                              maskimg))
        return masks
