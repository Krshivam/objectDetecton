import os
import sys
import cv2
import numpy as np 
import tensorflow as tf 

sys.path.append("..")
from utils import label_map_util
from utils import visualization_utils as vis_util 
MODEL_NAME = 'infernce_graph9'
VIDEO_NAME = 'test1.mp4'
d = {}
csv_file = open("report.csv",'w')
csv_file.write('image,class,score,x1,x2,y1,y2')
hitlim = 0.5
CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(CWD_PATH,'training1','labelmap.pbtxt')
PATH_TO_VIDEO = os.path.join(CWD_PATH,VIDEO_NAME)
NUM_CLASSES = 4
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map,max_num_classes=NUM_CLASSES,use_display_name=True)
category_index = label_map_util.create_category_index(categories)
detection_graph = tf.Graph()
with detection_graph.as_default():
	od_graph_def = tf.GraphDef()
	with tf.gfile.GFile(PATH_TO_CKPT,'rb') as fid:
		serialized_graph = fid.read()
		od_graph_def.ParseFromString(serialized_graph)
		tf.import_graph_def(od_graph_def,name='')
	sess = tf.Session(graph=detection_graph)

image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')
video = cv2.VideoCapture(PATH_TO_VIDEO)
while(video.isOpened()):
	ret,frame = video.read()
	temp_frame = frame
	frame_expanded = np.expand_dims(temp_frame,axis=0)
	(boxes,scores,classes,num) = sess.run([detection_boxes,detection_scores,detection_classes,num_detections],feed_dict={image_tensor:frame_expanded})
	nprehit = scores.shape[1]
	print(classes)
	vis_util.visualize_boxes_and_labels_on_image_array(temp_frame,np.squeeze(boxes),np.squeeze(classes).astype(np.int32),np.squeeze(scores),category_index,use_normalized_coordinates=True,line_thickness=4,min_score_thresh=0.80)

	#print (detection_classes)
	font = cv2.FONT_HERSHEY_SIMPLEX
	#cv2.putText(temp_frame,'Product_detected: ' + str(x),(5,30),font,0.8,(0,0xFF,0xFF),2,cv2.FONT_HERSHEY_SIMPLEX)
	cv2.imshow('object detector',temp_frame)
	if cv2.waitKey(1) == ord('q'):
		break
#print(len(boxes.shape))
hitf.flush()
hitf.close()
video.release()
cv2.destroyAllWindows()


