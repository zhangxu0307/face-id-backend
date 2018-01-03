# face-id-backend #

This is demo of face recoginition system backend. We heavily depend on some open source code, test on some implementations on benchmark dataset and finally we integrate them into our backend system.

Frontend is developed by Gao Zhongye.

## Requirements 

- Anaconda 5.0 Python 3.6.0
- Pytorch 0.2.0
- tensorflow 1.2.0
- keras 2.1.2
- openface 
- dlib 19.7.0
- django 1.11.7
- djangorestframework 3.7.3

## Code Structure 

The project is mainly based on Django project structure.

- face_algorithm: core face recognition algoritm
- face_id_backend: django configuration 
- face_recognition: django app
- media: face img and feature vector file
- test_json: json files, used in testing communication
- manage.py
- README.md

## face algorithm 

- center loss: untrained
- facenet tensorflow version: on testing
- joint bayes: train on openface and VGGface feature vector
- light cnn: on testing
- MTCNN keras version: conflict with TensorFlow 
- MTCNN pytorch version: implement with sphereface
- saim 2-channel CNN: untrained
- sphereface pytorch version: test on LFW successfully
- VGG face keras version: implement on django system, current best
- openface: implement on django system
- dlib align: implement on django system

## Dataset

- [LFW](http://vis-www.cs.umass.edu/lfw/)
- [Webface](http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html)

### LFW usage

-  [LFW pairs.txt解释](http://blog.csdn.net/zhongzhongzhen/article/details/78293789)
-  [LFW DataBase Accuracy 测定说明](http://blog.csdn.net/baidu_24281959/article/details/53218825)

## Usage

### Start Django service 
	
	cd face_id_backend
	python manage.py runserver 0.0.0.0：8888

### test on LFW
	
	cd face_id_backend/face_algoritm
	python lfw.py
you need to modify code and specify model you want to test, we will add argparser in the future.

## Reference

### Papers

- [A Discriminative Feature Learning Approach for Deep Face Recognition](https://ydwen.github.io/papers/WenECCV16.pdf)
- [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832)
- [Bayesian Face Revisited: A Joint Formulation](https://www.microsoft.com/en-us/research/wp-content/uploads/2012/01/JointBayesian.pdf)
- [A Light CNN for Deep Face Representation with Noisy Labels](https://arxiv.org/abs/1511.02683)
- [Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=2&cad=rja&uact=8&ved=0ahUKEwiG0aL8zrvYAhWBM5QKHZ0WDqYQFggvMAE&url=https%3A%2F%2Fkpzhang93.github.io%2FMTCNN_face_detection_alignment%2Fpaper%2Fspl.pdf&usg=AOvVaw1PCWFOy3q_C4vOFtrBjP-v)
- [SphereFace: Deep Hypersphere Embedding for Face Recognition](https://arxiv.org/abs/1704.08063)
- [Learning to Compare Image Patches via Convolutional Neural Networks](https://arxiv.org/abs/1504.03641)
- [Deep Face Recognition](http://www.robots.ox.ac.uk/~vgg/software/vgg_face/)
- [Deep Learning Face Representation from Predicting 10,000 Classes](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.646.8205&rep=rep1&type=pdf)
- [DeepID3: Face Recognition with Very Deep Neural Networks](https://arxiv.org/abs/1502.00873)
- [One Millisecond Face Alignment with an Ensemble of Regression Trees](https://pdfs.semanticscholar.org/d78b/6a5b0dcaa81b1faea5fb0000045a62513567.pdf)
- [Large-Margin Softmax Loss for Convolutional Neural Networks](https://arxiv.org/abs/1612.02295)
- [Siamese Neural Networks for One-shot Image Recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)
- [Naive-Deep Face Recognition: Touching the Limit of LFW Benchmark or Not?](https://arxiv.org/abs/1501.04690)

### Open source Code

- [facenet](https://github.com/davidsandberg/facenet)
- [VGGface](https://github.com/rcmalli/keras-vggface)
- [openface](https://github.com/cmusatyalab/openface)
- [Joint Bayes](https://github.com/cyh24/Joint-Bayesian)
- [light CNN](https://github.com/AlfredXiangWu/LightCNN)
- [MTCNN keras](https://github.com/xiangrufan/keras-mtcnn)
- [MTCNN pytorch](https://github.com/TropComplique/mtcnn-pytorch)
- [sphereface pytorch](https://github.com/clcarwin/sphereface_pytorch)

## Concat

[zhangxu0307@163.com](zhangxu0307@163.com)