# face id backend doc #

## 说明 ##
1. 本文档主要包含了当前face id后端系统中的使用说明，不含算法说明，具体算法参考README文档。
2. 后端需要配合前端使用。
3. 如果有需要修改数据库，参考django相关内容。

## 工程结构 ##
- doc 本说明文档目录
- face_algorithm 人脸识别算法目录，相关人脸识别算法训练和模型都在此目录下
- face_id_backend django工程目录
- face_recognition django app目录，是django向外提供服务的主要程序
- media 人脸图像储存目录
- test_json 测试base64通信的json文件目录，可忽略
- db.sqlite3 数据库文件，可迁移到mysql数据库
- manage.py django工程启动python文件
- README.md 项目说明文档

## 详细说明 ##

### face_algorithm ###

本目录下一级python文件都是已经训练好的模型算法程序，可以向外嵌入到django的工程中。

#### 人脸对齐模型 #####

- dtect_align.py 使用openface中的dlib模型对齐
- landmarks_mtcnn 使用mtcnn模型对齐检测 

#### 人脸识别模型 #####

- face_id.py open face模型
- vgg_face.py vgg face模型
- joint_bayes_face.py 联合贝叶斯模型，
- facenet_tf.py facenet 模型 tensorflow版本，测试效果不好，未集成到django
- light_cnn_pytorch.py lightcnn模型pytorch版本，测试效果良好，未集成到django
- light_cnn_tf.py lightcnn模型tensorflow版本，测试效果不好，未集成到django
- center_loss_face.py center loss模型，测试效果良好，未集成到django
- siam_CNN_face.py 孪生网络，未训练
- sphere_face_pt.py sphere face模型pytorch版本，测试效果良好，未集成到django

以上人脸识别基本流程为：
1. 预加载模型
2. getRep_XXX模型函数
	1. 调用上述对齐模型
	2. 深度网络前向过程，产生特征向量

#### 数据集相关 #####

此部分为数据集训练测试相关操作，与django系统应用无关

- webface.py webface相关操作
- lfw.py lfw相关操作

#### 特征向量的储存于操作 ####

- id_utils.py 

### face_id_backend ###

此目录下settings.py 配置有一些模型文件和人脸图像文件储存的默认路径，其余都为django自动生成的常规配置

    STATIC_URL = '/static/'

	CANDIDATEPATH = BASE_DIR+"/media/candidate_vec.pkl"
	IMAGEPATH = BASE_DIR+"/media/"
	files = os.listdir(IMAGEPATH)
	
	# 如果候选集路径，就加载，没有则新生成一个dataframe
	if  os.path.exists(CANDIDATEPATH):
	    #CANDIDATE = pd.read_pickle(CANDIDATEPATH)
	    CANDIDATE = loadFeatureVec(CANDIDATEPATH, format="pkl")
	else:
	    CANDIDATE = pd.DataFrame()
