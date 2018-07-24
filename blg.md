

在上一篇文章，介绍了词袋模型(Bag of Words)在图像检索中的应用。本文将简单的实现基于`BoW`的图像检索，包括以下内容：
- 基于OpenCV的实现，包括提取`sift`特征,聚类生产词典，建立kd-tree索引，检索。
- OpenCV类型`BOWTrainer`的使用
- 开源词袋模型DBow3的简单使用

### `BoW`的简单实现

`BoW`是将图像表示为`visual word`的直方图，也就是该图像包含每一个`visual word`的频数。`visual word`是图像库中具有代表性的特征向量，所以构建图像库`BoW`模型的步骤如下：
- 提取图像库中所有图像的特征，如：sifr,orb等
- 构建图像库的 `visual word`列表，也就是词典`vocabulary`。　这一步就是要提取图像库中的典型特征得到一个个`visual word`，通常是对上一步中提取到的图像特征进行聚类，得到的聚类中心就是`vocabulary`。
- 相对于`vocabulary`统计图像中特征对应于每个`visual word`的频数，