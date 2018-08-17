### Image Retrieval

在构建图像特征库的时候，通常不会使用原始的图像特征，这是由于Raw Feature有很多冗余信息，而且维度过高在构建特征数据库和匹配的时候效率较低。所以，通常要对提取到的原始特征进行重新编码。比较常用的三种编码方式：
- BoF , Bog of Feature 源于文本处理的词袋模型(Bog,Bag of Words)
- VLAD , Vector of Aggragate Locally Descriptor
- FV , fisher vector

### 特征提取
- [x] vlfeat sift 特征提取
- [x] PCA降维
- [x] kmeans聚类

### 编码方式
- [x] BoF
- [x] VLAD
- [ ] FV

### 源码结构说明
- DBow3 开源的词袋模型库
    - src 源代码
    - test 简单的使用DBow３的测试代码
- vlfeat 轻量级的视觉库
- src 图像检索的源代码

### version 0.1
上面一些基础的东西已经陆续实现了，接下来使用OpenCV实现一个简单的图像检索应用，包括：
- Vocabulary的创建、保存、加载
- 图像数据库的创建、保存、加载
- 图像的vlad表示
- 检索，返回类似的图片

>　这里只使用的VLAD表示图像，而没有测试BoW的原因是，电脑不行，BoW需要的Vocabulary的尺寸太大，训练时间太长。