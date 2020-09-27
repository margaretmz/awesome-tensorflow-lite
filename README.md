<p align="center">
    <img src="images/awesome-tflite.png" alt="awesome tflite">
</p>

# Awesome TFLite [![Awesome](https://awesome.re/badge.svg)](https://awesome.re) [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com) [![Twitter](https://img.shields.io/badge/Twitter-%40margaretmz-blue)](https://twitter.com/margaretmz)
TensorFlow Lite (TFLite) is a set of tools that help convert and optimize TensorFlow models to run on mobile and edge devices. It's currently running on more than 4 billion devices! With TensorFlow 2.x, you can train a model with tf.Keras, easily convert a model to .tflite and deploy it; or you can download a pretrained TFLite model from the model zoo.

This is an awesome list of TFLite models with sample apps, helpful tools and learning resources. This awesome list was created to:
* Showcase what the community has built with TFLite 
* Put all the samples side-by-side for easy reference
* Share knowledge and learning resources

Please submit a PR if you would like to contribute and follow the guidelines [here](CONTRIBUTING.md).

## Contents
* [What is new](#what-is-new)
* [Models with samples](#models-with-samples)
  * [Computer vision](#computer-vision)
  * [Text](#text)
  * [Speech](#speech)
* [Model zoo](#model-zoo)
  * [TFLite models](#tflite-models)
  * [TensorFlow models](#tensorflow-models) 
* [Ideas and Inspiration](#ideas-and-inspiration)
* [ML Kit examples](#ml-kit-examples)
* [Plugins and SDKs](#plugins-and-sdks)
* [Helpful links](#helpful-links)
* [Learning resources](#learning-resources)
  * [Documentation](#documentation)
  * [Blog posts](#blog-posts)
  * [Books](#books)
  * [Videos](#videos)
  * [Podcasts](#podcasts)
  * [MOOCs](#moocs)

## What is new
Here are the new features and tools of TensorFlow Lite:: <img src="images/icons/new.png" width='32' height='32'/>
* [Announcement of the new converter](https://groups.google.com/a/tensorflow.org/d/msg/tflite/Z_h7706dt8Q/sNrjPj4yGgAJ) - [MLIR](https://medium.com/tensorflow/mlir-a-new-intermediate-representation-and-compiler-framework-beba999ed18d)-based and enables conversion of new classes of models such as Mask R-CNN and Mobile BERT etc., supports functional control flow and better error handling during conversion. Enabled by default in the nightly builds\.
* [Android Support Library](https://github.com/tensorflow/tflite-support/tree/master/tensorflow_lite_support/java) - Makes mobile development easier ([Android](https://github.com/tensorflow/examples/blob/master/lite/examples/image_classification/android/EXPLORE_THE_CODE.md) sample code).
* [Model Maker](https://www.tensorflow.org/lite/guide/model_maker) - Create your custom [image & text](https://github.com/tensorflow/examples/tree/master/tensorflow_examples/lite/model_maker) classification models easily in a few lines of code. See below the Icon Classifier for a tutorial by the community.
* [On-device training](https://blog.tensorflow.org/2019/12/example-on-device-model-personalization.html) - It is finally here! Currently limited to transfer learning for image classification only but it's a great start. See the official [Android](https://github.com/tensorflow/examples/blob/master/lite/examples/model_personalization/README.md) sample code and another one from the community ([Blog](https://aqibsaeed.github.io/on-device-activity-recognition) | [Android](https://github.com/aqibsaeed/on-device-activity-recognition)).
* [Hexagon delegate](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/performance/hexagon_delegate.md) - How to use the TFLite Hexagon Delegate to speed up model inference on mobile and edge devices. Also see blog post  [Accelerating TensorFlow Lite on Qualcomm Hexagon DSPs](https://blog.tensorflow.org/2019/12/accelerating-tensorflow-lite-on-qualcomm.html).
* [Model Metadata](https://www.tensorflow.org/lite/convert/metadata) - Provides a standard for model descriptions which also enables [Code Gen and Android Studio ML Model Binding](https://www.tensorflow.org/lite/inference_with_metadata/codegen).

## Models with samples
Here are the TFLite models with app / device implementations, and references.  
Note: pretrained TFLite models from MediaPipe are included, which you can implement with or without MediaPipe. 

### Computer vision

| Task                            | Model                                                                                                                                                             | App \| Reference                                                                                                                                                                                                                                                                                                                                                                                                       | Source             |
| ------------------------------- |-------------------------------------------------------------------------------------------------------------------------------------------------------------------| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------       | -------------------|
| Classification                  | MobileNetV1 ([download](https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_1.0_224_quant_and_labels.zip))                          | [Android](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android) \| [iOS](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/ios) \| [Raspberry Pi](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/raspberry_pi) \| [Overview](https://www.tensorflow.org/lite/models/image_classification/overview) | tensorflow.org     |
| Classification                  | MobileNetV2                                                                                                                                                       | Recognize Flowers with TFLite on Android [Codelab](https://codelabs.developers.google.com/codelabs/recognize-flowers-with-tensorflow-on-android/#0) \| [Android](https://github.com/tensorflow/examples/tree/master/lite/codelabs/flower_classification/android)                                                                                                                                                       | TensorFlow team    |
| Classification                  | MobileNetV2                                                                                                                                                       | Skin Lesion Detection [Android](https://github.com/AakashKumarNain/skin_cancer_detection/tree/master/demo)                                                                                                                                                                                                                                                                                                             | Community          |
| Classification                  | EfficientNet-Lite0 ([download](https://github.com/margaretmz/icon-classifier/blob/master/ml-code/icons-50.tflite))                                                | Icon Classifier [Colab & Android](https://github.com/margaretmz/icon-classifier) \| [tutorial 1](https://medium.com/swlh/icon-classifier-with-tflite-model-maker-9263c0021f72) \| [tutorial 2](https://medium.com/@margaretmz/icon-classifier-android-app-1fc0b727f761)                                                                                                                                                | Community          |
| Object detection                | Quantized COCO SSD MobileNet v1 ([download](https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip)) | [Android](https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/android) \| [iOS](https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/ios) \| [Overview](https://www.tensorflow.org/lite/models/object_detection/overview#starter_model)                                                                                                                     | tensorflow.org     |
| Object detection                | YOLO                                                                                                                                                              | [Flutter](https://blog.francium.tech/real-time-object-detection-on-mobile-with-flutter-tensorflow-lite-and-yolo-android-part-a0042c9b62c6) \| [Paper](https://arxiv.org/abs/1506.02640)                                                                                                                                                                                                                                | Community          |
| Object detection                | MobileNetV2 SSD ([download](https://github.com/google/mediapipe/tree/master/mediapipe/models/ssdlite_object_detection.tflite))                                    | [Reference](https://github.com/google/mediapipe/blob/master/mediapipe/models/object_detection_saved_model/README.md)                                                                                                                                                                                                                                                                                                   | MediaPipe          |
| License Plate detection         | SSD MobileNet [(download)](https://github.com/ariG23498/Flutter-License/blob/master/assets/detect.tflite)                                                         | [Flutter](https://github.com/ariG23498/Flutter-License)                                                                                                                                                                                                                                                                                                                                                                | Community          |
| Face detection                  | BlazeFace ([download](https://github.com/google/mediapipe/tree/master/mediapipe/models/face_detection_front.tflite))                                              | [Paper](https://sites.google.com/corp/view/perception-cv4arvr/blazeface)                                                                                                                                                                                                                                                                                                                                               | MediaPipe          |
| Hand detection & tracking       | Palm detection & hand landmarks ([download](https://github.com/google/mediapipe/tree/master/mediapipe/models#hand-detection-and-tracking))                        | [Blog post](https://mediapipe.page.link/handgoogleaiblog) \| [Model card](https://mediapipe.page.link/handmc)                                                                                                                                                                                                                                                                                                          | MediaPipe          |
| Pose estimation                 | Posenet ([download](https://storage.googleapis.com/download.tensorflow.org/models/tflite/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite))             | [Android](https://github.com/tensorflow/examples/tree/master/lite/examples/posenet/android) \| [iOS](https://github.com/tensorflow/examples/tree/master/lite/examples/posenet/ios) \| [Overview](https://www.tensorflow.org/lite/models/pose_estimation/overview)                                                                                                                                                      | tensorflow.org     |
| Segmentation                    | DeepLab V3 ([download](https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/deeplabv3_257_mv_gpu.tflite))                                     | [Android & iOS](https://github.com/tensorflow/examples/tree/master/lite/examples/image_segmentation/) \| [Overview](https://www.tensorflow.org/lite/models/segmentation/overview) \| Flutter [Image](https://github.com/kshitizrimal/Flutter-TFLite-Image-Segmentation) \| [Realtime](https://github.com/kshitizrimal/tflite-realtime-flutter) \| [Paper](https://arxiv.org/abs/1706.05587)                            | tf.org & Community |
| Segmentation                    | Different variants of [DeepLab V3 models](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md) in TFLite                         |  Models on [TF Hub](https://tfhub.dev/s?module-type=image-segmentation&publisher=sayakpaul) with Colab Notebooks                                                                                                                                                                                                                                                                                                       | Community          |
| Hair Segmentation               | [Download](https://github.com/google/mediapipe/tree/master/mediapipe/models/hair_segmentation.tflite)                                                             | [Paper](https://sites.google.com/corp/view/perception-cv4arvr/hair-segmentation)                                                                                                                                                                                                                                                                                                                                       | MediaPipe          |
| Style transfer                  | [Arbitrary image stylization](https://github.com/tensorflow/magenta/tree/master/magenta/models/arbitrary_image_stylization)                                       | [Overview](https://www.tensorflow.org/lite/models/style_transfer/overview) \| [Android](https://github.com/tensorflow/examples/tree/master/lite/examples/style_transfer/android) \| [Flutter](https://github.com/PuzzleLeaf/flutter_tflite_style_transfer)                                                                                                                                                             | tf.org & Community |
| Style transfer                  | Better-quality style transfer models in TFLite                                                                                                                    |  Models on [TF Hub](https://tfhub.dev/sayakpaul/lite-model/arbitrary-image-stylization-inceptionv3/dr/predict/1) with Colab Notebooks                                                                                                                                                                                                                                                                                  | Community          |
| GANs                            | [U-GAT-IT](https://github.com/taki0112/UGATIT) (Selfie2Anime)                                                                                                     | [Project repo](https://github.com/margaretmz/selfie2anime-with-tflite) \| [Android](https://github.com/margaretmz/selfie2anime-with-tflite/tree/master/android) \| [Tutorial](https://medium.com/google-developer-experts/selfie2anime-with-tflite-part-1-overview-f97500800ffe)                                                                                                                                       | Community          |
| GANs                            | [White-box CartoonGAN](https://github.com/SystemErrorWang/White-box-Cartoonization) ([download](https://tfhub.dev/sayakpaul/lite-model/cartoongan/dr/1))          | [Project repo](https://github.com/margaretmz/Cartoonizer-with-TFLite) \| [Android](https://github.com/margaretmz/Cartoonizer-with-TFLite/tree/master/android) \| [Tutorial](https://blog.tensorflow.org/2020/09/how-to-create-cartoonizer-with-tf-lite.html)                                                                                                                                                           | Community          |
| Video Style Transfer            | Download: <br> [Dynamic range models](https://tfhub.dev/sayakpaul/lite-model/arbitrary-image-stylization-inceptionv3-dynamic-shapes/dr/transfer/1))               | [Android](https://github.com/farmaker47/video_style_transfer) \| [Tutorial](https://medium.com/@farmaker47/android-implementation-of-video-style-transfer-with-tensorflow-lite-models-9338a6d2a3ea)                                                                                                                                                                                                                    | Community          |

### Text
| Task                | Model                                                                                                                           | Sample apps                                                                                                                                                                                                                                       | Source             |
| ------------------- |---------------------------------------------------------------------------------------------------------------------------------| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------ |
| Question & Answer   | DistilBERT                                                                                                                      | [Android](https://github.com/huggingface/tflite-android-transformers/blob/master/bert)                                                                                                                                                            | Hugging Face       |
| Text Generation     | GPT-2 / DistilGPT2                                                                                                              | [Android](https://github.com/huggingface/tflite-android-transformers/blob/master/gpt2)                                                                                                                                                            | Hugging Face       |
| Text Classification | [Download](https://storage.googleapis.com/download.tensorflow.org/models/tflite/text_classification/text_classification.tflite) | [Android](https://github.com/tensorflow/examples/tree/master/lite/examples/text_classification/android) \|[iOS](https://github.com/khurram18/TextClassafication) \| [Flutter](https://github.com/am15h/tflite_flutter_plugin/tree/master/example) | tf.org & Community |

### Speech
| Task               | Model                              | App \| Reference                                                                      | Source       |
| ------------------ |------------------------------------| ------------------------------------------------------------------------------------- | ------------ |
| Speech Recognition | DeepSpeech                         | [Reference](https://github.com/mozilla/DeepSpeech/tree/master/native_client/java)     | Mozilla      |
| Speech Synthesis   | Tacotron-2, FastSpeech2, MB-Melgan | [Android](https://github.com/TensorSpeech/TensorflowTTS/tree/master/examples/android) | TensorSpeech |

<img src="images/icons/zoo.png" width='48' height='48'/>

## Model zoo 

### TFLite models
These are the TFLite models that could be implemented in apps and things:
* [MobileNet](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/README.md) - Pretrained MobileNet v2 and v3 models.
* TFLite models
  * [TFLite models](https://www.tensorflow.org/lite/models) - With official Android and iOS examples.
  * [Pretrained models](https://www.tensorflow.org/lite/guide/hosted_models) - Quantized and floating point variants.
  * [TensorFlow Hub](https://tfhub.dev/) - Set "Model format = TFLite" to find TFLite models.

### TensorFlow models
These are TensorFlow models that could be converted to TFLite and then implemented in apps and things:
* [TensorFlow models](https://github.com/tensorflow/models/tree/master/official) - Official TensorFlow models.
* [Tensorflow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) - Pre-trained on COCO, KITTI, AVA v2.1, iNaturalist Species datasets.

## Ideas and Inspiration
* [E2E TFLite Tutorials](https://github.com/ml-gde/e2e-tflite-tutorials) - Checkout this repo for sample app ideas and seeking help for your tutorial projects. Once a project gets completed, the links of the TFLite model(s), sample code and tutorial will be added to this awesome list.

<img src="images/icons/mlkit.png" width='32' height='32'/>

## ML Kit examples 
[ML Kit](https://developers.google.com/ml-kit) is a mobile SDK that brings Google's ML expertise to mobile developers.
* 10/1/2019 ML Kit Translate demo with material design - recognize, identify Language and translate text from live camera with ML Kit for Firebase - [Codelab](https://codelabs.developers.google.com/codelabs/mlkit-android-translate/#0) | [Android](https://github.com/googlecodelabs/mlkit-android/tree/master/translate) (Kotlin).
* 3/13/2019 Computer Vision with ML Kit - Flutter In Focus - [tutorial](https://youtu.be/ymyYUCrJnxU).
* 2/9/219 Flutter + MLKit: Business Card Mail Extractor - [tutorial](https://medium.com/flutter-community/flutter-mlkit-8039ec66b6a) | [Flutter](https://github.com/DaemonLoki/Business-Card-Mail-Extractor).
* 2/8/2019 From TensorFlow to ML Kit: Power your Android application with machine learning - [slides](https://speakerdeck.com/jinqian/from-tensorflow-to-ml-kit-power-your-android-application-with-machine-learning) | [Android](https://github.com/xebia-france/magritte) (Kotlin).
* 8/7/2018 Building a Custom Machine Learning Model on Android with TensorFlow Lite - [tutorial](https://medium.com/over-engineering/building-a-custom-machine-learning-model-on-android-with-tensorflow-lite-26447e53abf2).
* 7/20/2018 - ML Kit and Face Detection in Flutter - [tutorial](https://flatteredwithflutter.com/ml-kit-and-face-detection-in-flutter/).
* 7/27/2018 ML Kit on Android 4: Landmark Detection - [tutorial](https://medium.com/google-developer-experts/exploring-firebase-mlkit-on-android-landmark-detection-part-four-5e86b8deac3a).
* 7/28/2018 ML Kit on Android 3: Barcode Scanning - [tutorial](https://medium.com/google-developer-experts/exploring-firebase-mlkit-on-android-barcode-scanning-part-three-cc6f5921a108).
* 5/31/2018 ML Kit on Android 2: Face Detection - [tutorial](https://medium.com/google-developer-experts/exploring-firebase-mlkit-on-android-face-detection-part-two-de7e307c52e0).
* 5/22/2018 ML Kit on Android 1: Intro - [tutorial](https://medium.com/google-developer-experts/exploring-firebase-mlkit-on-android-introducing-mlkit-part-one-98fcfedbeee0).

## Plugins and SDKs
* [Edge Impulse](https://www.edgeimpulse.com/) - Created by [@EdgeImpulse](https://twitter.com/EdgeImpulse) to help you to train TFLite models for embedded devices in the cloud.
* [Fritz.ai](https://www.fritz.ai/) - An ML platform by [@fritzlabs](https://twitter.com/fritzlabs) that makes mobile development easier: with pre-trained ML models and end-to-end platform for building and deploying custom trained models.
* [MediaPipe](https://github.com/google/mediapipe) - A cross platform (mobile, desktop and Edge TPUs) AI pipeline by Google AI. (PM [Ming Yong](https://twitter.com/realmgyong)) | [MediaPipe examples](https://mediapipe.readthedocs.io/en/latest/examples.html).
* [Coral Edge TPU](https://coral.ai/) - Edge hardware by Google. [Coral Edge TPU examples](https://coral.ai/examples/).
* [TFLite Flutter Plugin](https://github.com/am15h/tflite_flutter_plugin/) - Provides a dart API similar to the TFLite Java API for accessing TensorFlow Lite interpreter and performing inference in flutter apps. [tflite_flutter on pub.dev](https://pub.dev/packages/tflite_flutter).

<img src="images/icons/helpful.png" width='32' height='32'/>

## Helpful links 
* [Netron](https://github.com/lutzroeder/netron) - A tool for visualizing models.
* [AI benchmark](http://ai-benchmark.com/tests.html) - A website for benchmarking computer vision models on smartphones.
* [Performance measurement](https://www.tensorflow.org/lite/performance/measurement) - How to measure model performance on Android and iOS.
* [Material design guidelines for ML](https://material.io/collections/machine-learning/patterns-for-machine-learning-powered-features.html) - How to design machine learning powered features. A good example: [ML Kit Showcase App](https://github.com/firebase/mlkit-material-android).
* [The People + AI Guide book](https://pair.withgoogle.com/) - Learn how to design human-centered AI products.
* [Adventures in TFLite](https://github.com/sayakpaul/Adventures-in-TensorFlow-Lite) - A repository showing non-trivial conversion processes and general explorations in TFLite.

## Learning resources 
Interested but not sure how to get started? Here are some learning resources that will help you whether you are a beginner or a practitioner in the field for a while.

<img src="images/icons/docs.png" width='32' height='32'/>

### Documentation 
* [TensorFlow Lite](https://www.tensorflow.org/lite)
* [TensorFlow Lite for Microcontrollers](https://www.tensorflow.org/lite/microcontrollers)

<img src="images/icons/blog.png" width='32' height='32'/>

### Blog posts 

* 4/20/2020 [What is new in TensorFlow Lite](https://blog.tensorflow.org/2020/04/whats-new-in-tensorflow-lite-from-devsummit-2020.html) - By Khanh LeViet.
* 4/17/2020 [Optimizing style transfer to run on mobile with TFLite](https://blog.tensorflow.org/2020/04/optimizing-style-transfer-to-run-on-mobile-with-tflite.html) - By Khanh LeViet and Luiz Gustavo Martins. 
* 4/14/2020 [How TensorFlow Lite helps you from prototype to product](https://blog.tensorflow.org/2020/04/how-tensorflow-lite-helps-you-from-prototype-to-product.html) -  By Khanh LeViet.
* 11/8/2019 [Getting  Started with ML on MCUs with TensorFlow](https://blog.particle.io/2019/11/08/particle-machine-learning-101/) -  By Brandon Satrom. 
* 8/5/2019 [TensorFlow Model Optimization Toolkit — float16 quantization halves model size](https://blog.tensorflow.org/2019/08/tensorflow-model-optimization-toolkit_5.html) - By the TensorFlow team.
* 7/13/2018 [Training and serving a real-time mobile object detector in 30 minutes with Cloud TPUs](https://blog.tensorflow.org/2018/07/training-and-serving-realtime-mobile-object-detector-cloud-tpus.html) - By Sara Robinson, Aakanksha Chowdhery, and Jonathan Huang.
* 6/11/2018 - [Why the Future of Machine Learning is Tiny](https://petewarden.com/2018/06/11/why-the-future-of-machine-learning-is-tiny/) - By Pete Warden. 
* 3/30/2018 - [Using TensorFlow Lite on Android](https://blog.tensorflow.org/2018/03/using-tensorflow-lite-on-android.html)) - By Laurence Moroney. 

<img src="images/icons/books.png" height='32'/>

### Books 
* 03/2020 Raspberry Pi for Computer Vision ([Complete Bundle](https://www.pyimagesearch.com/raspberry-pi-for-computer-vision) | [TOC](https://www.pyimagesearch.com/2019/04/05/table-of-contents-raspberry-pi-for-computer-vision/)) - By the PyImageSearch Team: Adrian Rosebrock ([@PyImageSearch](https://twitter.com/PyImageSearch)), David Hoffman, Asbhishek Thanki, Sayak Paul ([@RisingSayak](https://twitter.com/RisingSayak)), and David Mcduffee.
* 12/2019 [TinyML](http://shop.oreilly.com/product/0636920254508.do) - By Pete Warden ([@petewarden](https://twitter.com/petewarden)) and Daniel Situnayake ([@dansitu](https://twitter.com/dansitu)). 
* 10/2019 [Practical Deep Learning for Cloud, Mobile, and Edge](https://www.practicaldeeplearning.ai/) - By Anirudh Koul ([@AnirudhKoul](https://twitter.com/AnirudhKoul)), Siddha Ganju ([@SiddhaGanju](https://twitter.com/SiddhaGanju)), and Meher Kasam ([@MeherKasam](https://twitter.com/MeherKasam)).

<img src="images/icons/youtube.png" width='36' height='25'/>

### Videos 
* 7/25/2020 [Android ML by Hoi Lam](https://youtu.be/m_bEh8YifnQ) (GDG Kolkata meetup).
* 4/1/2020 [Easy on-device ML from prototype to production](https://youtu.be/ALxWJoh_BHw) (TF Dev Summit 2020).
* 3/11/2020 [TensorFlow Lite: ML for mobile and IoT devices](https://youtu.be/27Zx-4GOQA8) (TF Dev Summit 2020).
* 10/31/2019 [Keynote - TensorFlow Lite: ML for mobile and IoT devices](https://youtu.be/zjDGAiLqGk8).
* 10/31/2019 [TensorFlow Lite: Solution for running ML on-device](https://youtu.be/0SpZy7iouFU).
* 10/31/2019 [TensorFlow model optimization: Quantization and pruning](https://youtu.be/3JWRVx1OKQQ). 
* 10/29/2019 [Inside TensorFlow: TensorFlow Lite](https://youtu.be/gHN0jDbJz8E).
* 4/18/2018 [TensorFlow Lite for Android (Coding TensorFlow)](https://youtu.be/JnhW5tQ_7Vo).

### Podcasts
* 8/8/2020 [Talking Machine Learning with Hoi Lam](https://anchor.fm/talkingwithapples/episodes/Talking-Machine-Learning-with-Hoi-Lam-eiaj7v).

<img src="images/icons/mooc.png" width='32' height='32'/>

### MOOCs 
* [Introduction to TensorFlow Lite](https://www.udacity.com/course/intro-to-tensorflow-lite--ud190) - Udacity course by Daniel Situnayake (@dansitu), Paige Bailey ([@DynamicWebPaige](https://twitter.com/DynamicWebPaige)), and Juan Delgado.
* [Device-based Models with TensorFlow Lite](https://www.coursera.org/learn/device-based-models-tensorflow) - Coursera course by Laurence Moroney ([@lmoroney](https://twitter.com/lmoroney)).
