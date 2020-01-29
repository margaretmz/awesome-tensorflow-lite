# Awesome TFLite Collection
TensorFlow Lite is a set of tools that help convert TensorFlow models to run on edge devices. 

This is a collection of links to TFLite models along with sample app, model zoo, learning resources and helpful tools. Please submit a PR if you would like to contribute your TFLite models, demo apps or know of any other TFLite learning resources.

***
## TFLite models
TFLite models with app or device implementations: 

### Computer vision

| Task           | Model         | Implementation                         | Reference |
| -------------- |---------------| -------------------------------------- | --------- |
| Classification | MobileNetV1 ([download](https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_1.0_224_quant_and_labels.zip))| [Android](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android) \| [iOS](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/ios) \| [Raspberry Pi](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/raspberry_pi) |[tensorflow.org](https://www.tensorflow.org/lite/models/image_classification/overview)|
| Classification | MobileNetV2   | Skin Lesion Detection [Android](https://github.com/AakashKumarNain/skin_cancer_detection/tree/master/demo)|Community|
| Object detection | Quantized COCO SSD MobileNet v1 | [Model](https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip) \| [Android](https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/android) \| [iOS](https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/ios) |[tensorflow.org]       |
| Object detection | YOLO        | [Flutter](https://blog.francium.tech/real-time-object-detection-on-mobile-with-flutter-tensorflow-lite-and-yolo-android-part-a0042c9b62c6) \| [Paper](https://arxiv.org/abs/1506.02640)  | Community |
| Object detection | MobileNetV2 SSD  | [Model](https://github.com/google/mediapipe/tree/master/mediapipe/models/ssdlite_object_detection.tflite) \| [Reference](https://github.com/google/mediapipe/blob/master/mediapipe/models/object_detection_saved_model/README.md) | MediaPipe |
| Face detection | BlazeFace | [Model](https://github.com/google/mediapipe/tree/master/mediapipe/models/face_detection_front.tflite) \| [Paper](https://sites.google.com/corp/view/perception-cv4arvr/blazeface) \| [Model card](https://sites.google.com/corp/view/perception-cv4arvr/blazeface#h.p_21ojPZDx3cqq) | MediaPipe |

### Text

### Speech



