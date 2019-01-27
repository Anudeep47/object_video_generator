import numpy as np
import os
import six.moves.urllib as urllib
import tarfile
import tensorflow as tf
from distutils.version import StrictVersion

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
	raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'


# downloading model
opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
	file_name = os.path.basename(file.name)
	if 'frozen_inference_graph.pb' in file_name:
		tar_file.extract(file, os.getcwd())

# loading frozen model
detection_graph = tf.Graph()
with detection_graph.as_default():
	od_graph_def = tf.GraphDef()
	with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
		serialized_graph = fid.read()
		od_graph_def.ParseFromString(serialized_graph)
		tf.import_graph_def(od_graph_def, name='')

category_index = {
	1: {'id': 1, 'name': 'person'}, 
	2: {'id': 2, 'name': 'bicycle'}, 
	3: {'id': 3, 'name': 'car'}, 
	4: {'id': 4, 'name': 'motorcycle'}, 
	6: {'id': 6, 'name': 'bus'}, 
	8: {'id': 8, 'name': 'truck'}
}


def run_inference_for_single_image(image, graph):
	with graph.as_default():
		with tf.Session() as sess:
			ops = tf.get_default_graph().get_operations()
			all_tensor_names = {output.name for op in ops for output in op.outputs}
			tensor_dict = {}
			for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes', 'detection_masks']:
				tensor_name = key + ':0'
				if tensor_name in all_tensor_names:
					tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
	  		
			image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
			output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})
			output_dict['num_detections'] = int(output_dict['num_detections'][0])
			output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
			output_dict['detection_scores'] = output_dict['detection_scores'][0]

	return output_dict


def detect_objects_in_frame(image_np):
	image_np_expanded = np.expand_dims(image_np, axis=0)
	output_dict = run_inference_for_single_image(image_np, detection_graph)

	objects = set()
	for index, score in enumerate(output_dict['detection_scores']):
		if score > 0.2 and category_index.get(output_dict['detection_classes'][index]):
			objects.add(category_index[output_dict['detection_classes'][index]]['name'])
	return list(objects)
