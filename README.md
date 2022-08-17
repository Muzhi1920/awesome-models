# awesome-models

该仓库主要总结基于TensorFlow框架，实现的推荐模型。

- TensorFlow基础部分：主要包括Tensorflow的基础op、变量初始化和激活函数的选择、常见loss及其应用、各种FeatureColumn的内部实现，以及底层EmbeddingLookup细节曝光。

- 模型部分：主要聚焦于推荐系统的特征交互、序列推荐、召回模型、多目标模型的结构和融合排序思想。
排序、召回等模块常用的模型结构均以Jupyter实现以探究不同模型算法在矩阵运算的具体细节。

## Models List

|          Module           |          Model        | Paper
| :-----------------------: | :-------------------- | :---------|
| 特征交互 |  CAN | [CAN: Feature Co-Action for Click-Through Rate Prediction](https://arxiv.org/abs/2011.05625)|
| .. | DCN | [Deep & Cross Network for Ad Click Predictions](https://arxiv.org/pdf/1708.05123.pdf)|
| .. | PNN | [Product-based Neural Networks for User Response Prediction](https://arxiv.org/pdf/1611.00144.pdf) |
| .. | AFM | [Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks](https://arxiv.org/pdf/1708.04617.pdf)  |
| .. | DIN | [Deep Interest Network for Click-Through Rate Prediction](https://arxiv.org/pdf/1706.06978.pdf) |
| .. | FFM | [Field-aware Factorization Machines for CTR Prediction](http://dx.doi.org/10.1145/2959100.2959134)|
| .. |DeepFM|[DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](https://arxiv.org/abs/1703.04247) |
| .. |DeepCrossing| [Deep Crossing: Web-Scale Modeling without Manually Crafted Combinatorial Features](https://www.kdd.org/kdd2016/papers/files/adf0975-shanA.pdf)|
| .. | NCF | [Neural Collaborative Filtering](https://arxiv.org/pdf/1708.05031.pdf)|
| .. | NFM | [Neural Factorization Machines for Sparse Predictive Analytics](https://arxiv.org/abs/1708.05027)  |
|多目标结构| MMoE| [Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts](https://dl.acm.org/doi/10.1145/3219819.3220007)  |
| .. | PLE    | [Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations](https://dl.acm.org/doi/10.1145/3383313.3412236)  |
|序列与召回|ComiRec | [Controllable Multi-Interest Framework for Recommendation](https://dl.acm.org/doi/abs/10.1145/3394486.3403344)  |
| .. | STAMP | [STAMP: Short-Term Attention/Memory Priority Model for Session-based Recommendation](https://dl.acm.org/doi/10.1145/3219819.3219950)|
| .. | SASRec    | [Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781)  |

## 目录
- TensorFlow基础
- 特征交互
- 召回与序列推荐
- 多目标结构

## 知乎与博客
- 个人知乎专栏：[《推荐系统实践》](https://www.zhihu.com/column/c_1432753427968999424)
- 个人Blog：https://11010101.xyz/
