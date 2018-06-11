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
- raw_face_img 原始人类图像数据，可以在此文件夹下批量加入数据库，文件组织形式为人脸的jpg或png格式图片，图片文件名为学号+空格+姓名+文件类型后缀名的格式

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
- <del> facenet_tf.py facenet 模型 tensorflow版本，测试效果不好，未集成到django </del> 
- <del> light_cnn_pytorch.py lightcnn模型pytorch版本，测试效果良好，未集成到django</del>
- <del> light_cnn_tf.py lightcnn模型tensorflow版本，测试效果不好，未集成到django</del>
- <del> center_loss_face.py center loss模型，测试效果良好，未集成到django</del>
- <del> siam_CNN_face.py 孪生网络，未训练</del>
- sphere_face_pt.py sphere face模型pytorch版本，测试效果良好，未集成到django

以上人脸识别基本流程为：
1. 预加载模型
2. getRep_XXX模型函数
	1. 调用上述对齐模型
	2. 深度网络前向过程，产生特征向量，并返回

#### 数据集相关 #####

此部分为数据集训练测试相关操作，与django系统应用无关
为减少工程代码的冗余和强耦合性，此部分单独集成至face_algorithm_explore project中，具体github链接在[https://github.com/zhangxu0307/face_algorithm_explore](https://github.com/zhangxu0307/face_algorithm_explore "face_algorithm_explore")

- <del>webface.py webface相关操作</del>
- <del>lfw.py lfw相关操作</del>

#### 特征向量的储存于操作 ####

- id_utils.py 

### face_id_backend ###

此目录下settings.py 配置有一些模型文件和人脸图像文件储存的默认路径以及两个重要的算法参数，其余都为django自动生成的常规配置

	STATIC_URL = '/static/'
	METHOD = "VGGface"  # 人脸算法选择
	CANDIDATEPATH = BASE_DIR + "/media/candidate_vec.pkl"  # 候选特征向量储存文件
	IMAGEPATH = BASE_DIR + "/media/"  # 候选人人脸图片目录
	files = os.listdir(IMAGEPATH)
	RAWFACEIMGPATH = BASE_DIR+"/raw_face_img/"  # 需要批量加入的候选人人脸图片
	
	print("system basic config info....")
	print("CANDIDATEPATH:", CANDIDATEPATH)
	print("IMAGEPATH:", IMAGEPATH)
	print("RAWFACEIMGPATH:", RAWFACEIMGPATH)
	
	# 如果候选集路径，就加载，没有则新生成一个dataframe
	if os.path.exists(CANDIDATEPATH):
	    CANDIDATE = loadFeatureVec(CANDIDATEPATH, format="pkl")
	else:
	    CANDIDATE = pd.DataFrame()
	
	# 算法中两个关键性阈值
	SIMILARITY_THRESHOLD = 0.6  # 相似度阈值
	JOINT_BAYES_THRESHOLD = 50  # joint bayes的阈值

### 切换算法的方法 ###

目前本系统可以支持两种人脸识别算法，分别是sphereface和VGGface，具体方法需要再settings.py中指定即可。然而由于存在非常神奇的兼容性问题，其他py文件某些部分也需要做相应改动。改动之处在：

1. face_algorithm/id_utils.py 中的第一行
2. face_recognition/views.py 中的第一行

以上两部分请按照注释的要求做相应改动即可。

### face_recognition ###

此目录是django工程的app目录，是django主要的服务端程序

- my_serializers.py 定制的序列化器，用来通信
- serializers.py django-rest的序列化器，已废弃
- urls.py url文件
- views.py 中间逻辑层，实现功能的主要代码均在这个文件下，具体实现的API见下文。

## API接口文档 ##

- http://114.212.84.6:8888/api/recognition
	- 功能：人脸识别 
	- 方法：POST
	- 参数：json，包括picture、boundingbox、threshold三个字段
	- 返回： 人脸图像、学号、姓名、相似度

- http://114.212.84.6:8888/api/register/
	- 功能：从相机拍摄注册
	- 方法：POST
	- 参数：json，包括picture、ID、name等字段
	- 返回： "detail": "new face has been saved!"
	
- http://114.212.84.6:8888/api/delete/
	- 功能：删除指定单个人的记录
	- 方法：POST
	- 参数：delete_ID:学号
	- 返回： "detail": "delete success!"
- http://114.212.84.6:8888/api/register_batch/
	- 功能：从文件中批量构建人脸数据库信息
	- 方法：POST
	- 参数：无
	- 返回： "detail": "all face has been saved!"
	
- http://114.212.84.6:8888/api/clear/
	- 功能：清除所有数据信息
	- 方法：POST
	- 参数：无
	- 返回： "detail": "all data has been cleaned!"


## 阈值修改 ##

- SIMILARITY_THRESHOLD 余弦相似度阈值，范围在-1~1之间，大部分分布于0~1之间，是召回最相似的人的检查标准。
- JOINT_BAYES_THRESHOLD 联合贝叶斯阈值，两两验证阈值，如果使用VGGface模型，一般本人joint bayes得分会在250以上，非本人得分一般在100左右，理想情况是负值。如果使用sphereface模型，一般本人会在70-100之间，不是本人一般在-20~20之间.