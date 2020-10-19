### To Do

* badcase->transformer
* 损失函数：label smooth，focal，triple
* 数据增强：resize，random resize crop，blur
* 优化器：sgd+pleatu，cos scheduler
* 辅助数据：yolo目标检测，姿态检测
* 模型：resnest，se_resnext，wsdan，efficient

### 10.13

* 训练了senet，resnest
* 测试了cbam crop7
* cbam dif resize ***9967***
* 测试了优化策略cos和reduce + adam

### 10.12

* 换了一块服务器
* 由于服务器大小，重新训练了一批数据
* 训练了wsdan
* 测试了crop
* 差分lr

### 10.11

* 服务器满了，半天训练白给
* 固定了种子测试了cbam base，一个结果提交了，***99.60***，其他还在训练中
* 固定了种子基于res18测试了transformer，结果是resize效果最好，之后打算试试resize+crop

### 10.10

* reswsl16d太大了
* 填充缩放比原缩放效果好一点点；高斯模糊效果稍微低一点点；random crop效果很差
* 提交了修改后的base cbam和resize cbam，结果都不好，决定固定随机种子之后再重新训练一下看看

### 10.9

* 调整了dataloader里图片长和宽搞反的问题，开始测试transforemr效果

### 10.8

* 用最基本的配置（10.6）分成更加细的lr来训练wsl，cbam，res18
* 提交了resnet cbam最好的结果，b=8，lr=1e-5，mAP=0.93，提交结果***99.60***

### 10.7

* 提交了resnet18的训练最好结果，b=8，lr=3e-5，mAP=0.898，提交结果***98.72***
* 训练了resnext101和resnextwsl

### 10.6

* 改正了读取数据集的bug，用最基本adam+crossentropy测试了基本代码
* 成功提交了数据，提交了resnet18，batchsize=8，lr=1e-4的第一个epoch的结果，mAP=0.72，提交结果***97.95***