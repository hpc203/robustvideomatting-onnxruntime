# robustvideomatting-onnxruntime
使用ONNXRuntime部署鲁棒性视频抠图，包含C++和Python两种版本的程序

在PeterL1n的github仓库里有一套很火的开源代码BackgroundMattingV2，在这个算法里，有两个输入，
一个是带人物的图片，另一个是不带人物的纯背景图片，有了背景作为一个很强的先验指导，使得BMV2在4K高分率的图片上以很快的性能跑出较高的抠图精度，
但是对于用户来讲，这是一个很不友好的。因为用户在实际拍摄时很少会刻意再去拍一个背景图，这就导致BackgroundMattingV2的应用场景很受限。
不过在最近，PeterL1n发布了一套新的视频抠图开源程序，这时的输入只有一张RGB图，这个就很好。因此我决定编写一套视频抠图的推理程序。
起初，我想使用opencv的dnn模块作为推理引擎，但是程序运行到cv2.cv2.dnn.readNet(modelpath) 这里时报错，因此我决定使用onnxruntime
作为推理引擎，程序能正常运行

模型文件在百度网盘，链接：https://pan.baidu.com/s/1dnbmmjrztex64xDjMzHSUA 
提取码：616v

下载完模型文件后，就可以运行程序里，C++版本的主程序是main.cpp，Python版本的主程序是main.py
