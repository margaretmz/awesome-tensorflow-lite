# Awesome TFLite Collection
TensorFlow Lite is a set of tools that help convert TensorFlow models to run on edge devices. 

This is a collection of links to TFLite models along with sample app, model zoo, learning resources and helpful tools. Please submit a PR if you would like to contribute your TFLite models, demo apps or know of any other TFLite learning resources.

***
## TFLite models
Here are the TFLite models with app / device implementations, and references.
Note: pretrained TFLite models from MediaPipe are included, which you can implement with or without MediaPipe. 

### Computer vision

| Task           | Model         | App \| Reference                        | Source |
| -------------- |---------------| -------------------------------------- | --------- |
| Classification | MobileNetV1 ([download](https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_1.0_224_quant_and_labels.zip))| [Android](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android) \| [iOS](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/ios) \| [Raspberry Pi](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/raspberry_pi) \| [Overview](https://www.tensorflow.org/lite/models/image_classification/overview)| tensorflow.org |
| Classification | MobileNetV2   | Skin Lesion Detection [Android](https://github.com/AakashKumarNain/skin_cancer_detection/tree/master/demo)|Community|
| Object detection | Quantized COCO SSD MobileNet v1 ([download](https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip)) | [Android](https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/android) \| [iOS](https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/ios) \| [Overview](https://www.tensorflow.org/lite/models/object_detection/overview#starter_model) | tensorflow.org |
| Object detection | YOLO        | [Flutter](https://blog.francium.tech/real-time-object-detection-on-mobile-with-flutter-tensorflow-lite-and-yolo-android-part-a0042c9b62c6) \| [Paper](https://arxiv.org/abs/1506.02640)  | Community |
| Object detection | MobileNetV2 SSD ([download](https://github.com/google/mediapipe/tree/master/mediapipe/models/ssdlite_object_detection.tflite)) | [Reference](https://github.com/google/mediapipe/blob/master/mediapipe/models/object_detection_saved_model/README.md) | MediaPipe |
| Face detection | BlazeFace ([Model](https://github.com/google/mediapipe/tree/master/mediapipe/models/face_detection_front.tflite)) | [Paper](https://sites.google.com/corp/view/perception-cv4arvr/blazeface) \| [Model card](https://sites.google.com/corp/view/perception-cv4arvr/blazeface#h.p_21ojPZDx3cqq) | MediaPipe |
|Hand detection & tracking | Download: [Palm detection](https://github.com/google/mediapipe/tree/master/mediapipe/models/palm_detection.tflite), [2D hand landmark](https://github.com/google/mediapipe/tree/master/mediapipe/models/hand_landmark.tflite), [3D hand landmark](https://github.com/google/mediapipe/tree/master/mediapipe/models/hand_landmark_3d.tflite) | [Blog post](https://mediapipe.page.link/handgoogleaiblog) \| [Model card](https://mediapipe.page.link/handmc) | MediaPipe |
| Pose estimation | Posenet ([download](https://storage.googleapis.com/download.tensorflow.org/models/tflite/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite)) | [Android](https://github.com/tensorflow/examples/tree/master/lite/examples/posenet/android) \| [Overview](https://www.tensorflow.org/lite/models/pose_estimation/overview)| tensorflow.org |
| Segmentation | DeepLab V3 ([Download](https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/deeplabv3_257_mv_gpu.tflite)) |  [Flutter](https://github.com/kshitizrimal/Flutter-TFLite-Image-Segmentation) \| [Paper](https://arxiv.org/abs/1706.05587) | Community | 
| Segmentation | DeepLab V3 ([Download](https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/deeplabv3_257_mv_gpu.tflite)) | [Android](https://github.com/tensorflow/examples/tree/master/lite/examples/image_segmentation/android) \| [iOS](https://github.com/tensorflow/examples/tree/master/lite/examples/image_segmentation/ios) \| [Overview](https://www.tensorflow.org/lite/models/style_transfer/overview)  | tensorflow.org |
| Hair Segmentation | [download](https://github.com/google/mediapipe/tree/master/mediapipe/models/hair_segmentation.tflite) | [Paper](https://sites.google.com/corp/view/perception-cv4arvr/hair-segmentation) \| [Model card](https://sites.google.com/corp/view/perception-cv4arvr/hair-segmentation#h.p_NimuO7PgHxlY) | MediaPipe | 
| Style transfer |  Download: [Style prediction](https://storage.googleapis.com/download.tensorflow.org/models/tflite/arbitrary_style_transfer/style_predict_quantized_256.tflite) & [Style transform](https://storage.googleapis.com/download.tensorflow.org/models/tflite/arbitrary_style_transfer/style_transfer_quantized_dynamic.tflite) | [Overview](https://www.tensorflow.org/lite/models/style_transfer/overview) | tensorflow.org |
|  |  |  |  | 

### Text
| Task           | Model         | App \| Reference                       | Source    |
| -------------- |---------------| -------------------------------------- | --------- |
| Question & Answer | DistilBERT | [Android](https://github.com/huggingface/tflite-android-transformers/blob/master/bert) | Hugging Face | 
| Text Generation | GPT-2 / DistilGPT2 | [Android](https://github.com/huggingface/tflite-android-transformers/blob/master/gpt2) | Hugging Face | 
### Speech
| Task               | Model    | App \| Reference    | Source    |
| ------------------ |----------| ------------------- | --------- |
| Speech Recognition | DeepSpeech | [Reference](https://github.com/mozilla/DeepSpeech/tree/master/native_client/java) | Mozilla | 

## ML Kit examples



