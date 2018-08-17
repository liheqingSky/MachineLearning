# MachineLearning
#0、准备文件
# deploy.prototxt： 网络结构配置文件
# bvlc_alexnet.caffemodel： 网络权重文件
# 测试图像jpg

#1、加载网络
caffe_root = '../../'

# 网络实施结构配置文件、及参数（权重）文件
deploy = caffe_root + 'models/bvlc_alexnet/deploy.prototxt'
caffemodel = caffe_root + 'models/bvlc_alexnet/bvlc_alexnet.caffemodel'
net = caffe.Net(deploy,  # 定义模型结构 
          caffemodel,  # 包含了模型的训练权值
          caffe.TEST)  # 使用测试模式(不执行dropout)
          
#2、测试图像预处理：减均值、调整大小
# 2.1 加载ImageNet图像均值 
# mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
# mu = mu.mean(1).mean(1)  # 对所有像素值取平均以此获取BGR的均值像素值

# 2.2 图像预处理
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))    # 转换图像维度 28x28x3---> 3x28x28
#transformer.set_mean('data', mu)              # 加载均值
#transformer.set_raw_scale('data', 255)        # 缩放至[0， 255]
transformer.set_channel_swap('data', (2,1,0)) # RGB格式转换为BGR格式

#3、运行网络：
#3.1 导入输入数据
im = caffe.io.load_image(img)                         # 加载图像
net.blobs['data'].data[...] = transformer.preprocess('data', im)  # 导入输入图像

#3.2 执行测试
start = time.clock()
net.forward()   # 执行测试
end = time.clock()
print('classification time: %f s' % (end - start))

#4、查看分类结果
# 用于图像分类的网络的最后一层一般是一个名为’prob’的SoftMax网络，这个名为’prob’层的输出即为反应该图像在各分类下的概率向量。
# 而prob层的输出Blob也名为prob
category = net.blobs['prob'].data[0].argmax() # 最大概率的分类

# 保存分类结果写文件
with open(./prob.txt, "w+") as f:
  out_data = net.blobs['prob'].data
  for num in out_data.flat:
    fwrite(str(num))
 


