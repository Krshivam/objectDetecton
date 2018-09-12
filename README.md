# object_detecton_tensorflow
OS used - Debian \
tensorflow without GPU \
Pyhton V2
Assume Current working directory as root \
Install Python\
Open terminal and type pip install --upgrade tensorflow \
Download the tensorflow object detection API either via CLI or download option (detailed below for each one)\
Link - https://github.com/tensorflow/models \
For CLI - open terminal and type git clone https://github.com/tensorflow/models \
Then a new folder models will be created. \
Download the Faster-RCNN-Inception-V2-COCO model from TensorFlow's model zoo (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)\
Extract the *.tar.gz file in /root/models/research/object_detection folder \
Download this repository and extract all contents in /root/models/research/object_detection folder \
Compile protobufs (below is the command copy and paste it in terminal but first cd - models/research) \
Copy ans paste   protoc --python_out=. ./object_detection/protos/anchor_generator.proto ./object_detection/protos/argmax_matcher.proto ./object_detection/protos/bipartite_matcher.proto ./object_detection/protos/box_coder.proto ./object_detection/protos/box_predictor.proto ./object_detection/protos/eval.proto ./object_detection/protos/faster_rcnn.proto ./object_detection/protos/faster_rcnn_box_coder.proto ./object_detection/protos/grid_anchor_generator.proto ./object_detection/protos/hyperparams.proto ./object_detection/protos/image_resizer.proto ./object_detection/protos/input_reader.proto ./object_detection/protos/losses.proto ./object_detection/protos/matcher.proto ./object_detection/protos/mean_stddev_box_coder.proto ./object_detection/protos/model.proto ./object_detection/protos/optimizer.proto ./object_detection/protos/pipeline.proto ./object_detection/protos/post_processing.proto ./object_detection/protos/preprocessor.proto ./object_detection/protos/region_similarity_calculator.proto ./object_detection/protos/square_box_coder.proto ./object_detection/protos/ssd.proto ./object_detection/protos/ssd_anchor_generator.proto ./object_detection/protos/string_int_label_map.proto ./object_detection/protos/train.proto ./object_detection/protos/keypoint_box_coder.proto ./object_detection/protos/multiscale_anchor_generator.proto ./object_detection/protos/graph_rewriter.proto\
In the same directory (i.e /models/research) run python setup.py build (via CLI)\
Then run python setup.py install \
Now cd to root\
cd to /root/models/research/object_detection\
#Run the training
copy and paste it in terminal python legacy/train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_pets.config\
#Export inference graph
Using the command given below, XXXX in model.ckpt-XXXX will be replaced with the highest-numbered .ckpt file in the training folder:\
python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_pets.config --trained_checkpoint_prefix training/model.ckpt-XXXX --output_directory inference_graph
Now run python img_detection.py to detect images\
run python video_detection.py to detect in videos\
run python webcam_detect.py to detect using webcam













