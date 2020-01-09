#人脸识别步骤：
#前提：编译好Caffe环境 #本地caffe安装路径：/home/cy/CodeDemo/caffe
Step1:准备人脸识别的正负样本数据集，其格式如下：
Step2:通过face-lmdb.sh脚本，制作能用于Caffe的LMDB格式的数据源  
Step3:准备网络结构模型(MN_train.prototxt)和超参数文件(MN_train_solver.prototxt)
Step4:通过train.sh脚本开始训练，生成MN_train_solver_iter_20000.caffemodel和MN_train_solver_iter_20000.solverstate文件 
Step5:准备MY_MN_Deploy.prototxt文件和MY_MN_full_deploy.prototxt文件，为了实现多尺度人脸检测，我们需要将我们的MobileNet网络结构中最后的全连接层换成卷积层
Step6:通过innerToConv2.py脚本后生成我们最终的网络结构文件MY_MN_full_conv.caffemodel   
Step7:最后的检测效果图如下：

#############
#本地数据集存储路径：/mnt/faceDataset/train15738x15663和/mnt/faceDataset/test2079x2066
#本地lmdb数据路径：/home/cy/CodeDemo/faceDetect/face_train_lmdb
#prototxt文件路径：/home/cy/CodeDemo/faceDetect
#相关模型路径：/home/cy/CodeDemo/faceDetect/model
#shell/python脚本路径：/home/cy/CodeDemo/faceDetect/