# DriverLicense
驾驶证识别项目

### 背景介绍

由于能够提供的驾驶证数据非常少（总共不到500张），因此需要写脚本去生成虚拟数据，本项目花了大量的时间在数据制造上。该项目是典型的基于深度学习的OCR，其中百度、阿里、旷视、美团等巨头均有该技术的成熟解决方案。基于深度学习的OCR技术通用思路是：首先定位出证件的位置，然后用方向模型识别驾驶证朝向（四个方向），在用文本检测网络定位文本条目的具体位置，最后将定位出的文本条目截取出来，送到文本序列识别网络进行识别，文本序列识别网络技术目前基本采用CRNN的架构。  
本项目采用PSENET（文本定位）+ CRNN（文本识别）进行识别。

### 相关库

* pytorch
* opencv3
* PIL

### 数据生成

数据生成采用背景模板+数据填充的方式，背景模板是网上找的驾驶证通用背景，文本填充采用PIL库的文本绘制方法。数据制造在旋转、模糊程度、字体、粗细、滤波、透视变换等方面做了大量工作，能够产生各种各样的驾驶证数据。在文字信息方面，身份证信息完全根据真实规则随机生成；所有日期根据实际年月份在合理的范围内随机生成；姓名根据全国姓氏统计及常用名称随机组合生成了一半的数据，另一半数据根据常用的5000个中文随机生成；地址数据根据中国省、市、县、街道的数据库随机抽取生成。  

### 文本检测

具体参考PSENET论文及网络上相关解析文章

### 文本识别

具体参考CRNN论文及网络上相关解析文章。  
注：CRNN做中文识别需要大量数据，该项目一共生成了约200W张文本条目。

### 注意事项
CRNN和PSENET模型最好在linux系统中训练，但是可以在windows系统中调用

### Demo界面软件（Release文件夹）
运行Demo之前，需要下载模型文件，放置到weights文件夹下面。
demo软件运行位置：release/app.py  
依赖项：release/requirements.txt  
运行命令：  
```Command
    python app.py
```  

软件操作说明：release/doc文件夹下面

### 脚本说明
* CRNN
CRNN确定词条内容，会输出一系列识别出来的词条，但是不一定连续，有可能有遗漏。
	* Train.py: 训练代码

* PSENET
用psenet定位词条（可选定位纯中文或中英文，中英文更准，但是更慢）
	* train_ctw1500.py：用于训练PSENET模型。训练前需要调psenet/dataset/ctw1500_loader.py里的训练路径

* Datagen
用于生成假驾驶证图片集来训练模型
	* fake_id_generator： 生成身份证号
	* Material：中文数据集
	* data_generator.py：核心代码，包含基本所有生成代码
	* data_generator _for_angle.py： 默认参数与data_generator.py不一样
	* data_generator_faster_rcnn.py： 默认参数与data_generator.py不一样，已经不用
	* faker.py：生成驾驶证中不同的文字信息
	* angle_detect.py：通过angle_finder模型判断驾驶证方向（四个分类，上下左右）。模型选用开源证件方向判别模型 https://github.com/chineseocr/chineseocr
	* build_PSENET_data.py: 将data_generator.py生成的数据根据PSENET要求的目录格式摆放
	* chinese_char_count.py：中文词频统计
	* cl_build_train_txt.py： 根据cut的词条图片及图片名称，生成一个CRNN训练数据集
	* ctw_format_validation.py： 可视化验证label_to_ctw_format.py转换后的标注数据是否准确
	* label_to_ctw_format.py：本脚本用于将data_generator.py生成的标注转化为14点标注格式
	* remove_bad_template.py： 移除坏的模板
	* run.py： 将图片存放到mymodel.h5文件中，隐藏图片
	* to_lmdb.py： 将图片、标注打包到一个数据库中
	* add_face.py：向驾驶证模板上贴照片
	* address_5_class.py：以真实省市乡镇街道数据生成假地址
	* chinese_name.py：生成中文名
	* gaoqing_to_diqing.py：高清图转成低清图
	* label_code.py：将label（json）信息转成txt
	* perspective_test.py：对图片进行透视变换
	* popular_name.py：生成最常见姓氏和名字txt文件
	* remove_bad_background.py：移除不好的背景：1.宽度小于长度的（width<height);2.尺寸过小的
	* rm_label_if_not_match_a_image.py：删除找不到对应图片的label
	* test_template.py：实验不同的图片处理效果

* Release
	* demo.py：软件包含了调用PSENET/CRNN以及后处理的代码，从PSENET和CRNN中将推理部分的代码抽离出来。模型依赖Pytorch深度学习框架，使用PyQt5界面