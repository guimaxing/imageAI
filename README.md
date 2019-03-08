IMAGEAI
目录结构

|--iamgeAI/     
|   |
|   |--Face_acquisition_from_video/       #人脸获取
|   |   |
|  	|   |--data/
|   |   |   |
|   |   |   |--data_dlib/* 预训练文件 shape_predictor_68_face_landmarks.dat、dlib_face_recognition_resnet_model_v1.dat、shape_predictor_5_face_landmarks.dat
|   |   |   |
|   |   |   |--data_faces_from_camera/    #存储获取的图片
|   |   |   
|   |   |-- face_acquisition.py           #程序运行：python face_acquisition.py，输入录入对象
|   |   
|   |--Face_Recognition/                  #人脸识别     
|   |   |
|   |   |--knn_data/
|   |   |   |
|   |   |   |--train/                     #存储knn模型人脸数据
|   |   |   |
|   |   |   |--test/                      #测试
|   |   |   
|   |   |--model/                         #存储knn模型
|   |   |
|   |   |--xml/                           #cv2 目标识别参数文件
|   |   |
|   |   |--face_recognition_knn.py        #knn模型建立，  python face_recognition_knn.py
|   |   |
|   |   |--facerec_from_webcam.py --------|  
|   |   |                                 |--#调用摄像头，实时人脸识别
|   |   |--facerec_from_webcam_faster.py--|
|   |   
|   |--object_detection/   
|   |   |
|   |   |--keras-yolo3/
|   |   |   |
|   |   |   |--real_time_video_person_detect.py   #对象识别，实时监控
|   |   |   |
|   |   |   |--object_detection_yolo_for_train.py #人物识别，截图保存
|   |   |    
|   |   |相关模型由于太大上传不了，见附件链接信息   
|   |   |    
|   |       
|   |--safety_helmet_recognition/      
|   |   |    
|   |   |--train/   #训练集    
|   |   |--val/     #验证集  
|   |   |   
|   |   |--train.py  #python train.py    
|   |   |
|   |   |--predict.py #预测代码
|   |   |
|   |   |

图片均来源于网上

1、人脸识别
face_recognition_knn.py   对图片进行特征提取，建立knn模型

facerec_from_webcam_faster.py     调取笔记本摄像头，调用knn模型，实现实时人脸人脸（也可以修改为通过图片的输入进行人脸识别）
eg:
facerec_from_webcam 和 facerec_from_webcam_faster 区别在于获取图像的大小

2、图像识别 
object_detection_yolo_for_train.py   生成训练数据   识别图片：带安全帽和不带安全帽，标注数据集
存储格式为： ./train/no_hat/ ; ./train/has_hat/ ;
			 ./val/no_hat/ ; ./val/has_hat/ ;

3、图像分类模型训练
safety_helmet_recognition/train.py
使用vgg或者resnet预训练模型，提取网络结构，输入模型训练，并生成分类模型

4、以上介绍的各项功能都以是单独功能模块，组合起来可以实现人脸识别，目标检测，特定业务的目标检测等。

所用相关模型文件可以通过下面百度网盘链接进行下载：
链接：https://pan.baidu.com/s/1VMqkFPoblt2xf47cLgAl6Q 
提取码：wwma 

