

在上一篇文章，介绍了词袋模型(Bag of Words)在图像检索中的应用。本文将简单的实现基于`BoW`的图像检索，包括以下内容：
- 基于OpenCV的实现，包括提取`sift`特征,聚类生产词典，建立kd-tree索引，检索。
- OpenCV类型`BOWTrainer`的使用
- 开源词袋模型DBow3的简单使用

### `BoW`的简单实现

`BoW`是将图像表示为`visual word`的直方图，也就是该图像包含每一个`visual word`的频数。