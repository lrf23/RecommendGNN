# 机器学习大作业

### 模型与框架介绍

+ [HCCF模型步骤](https://github.com/lrf23/RecommendGNN/blob/main/HCCF%E6%A8%A1%E5%9E%8B%E6%AD%A5%E9%AA%A4.md) 介绍了HCCF和Rechorus框架进行模型训练的基本流程。
+ [Rechorus框架模型分类](https://github.com/lrf23/RecommendGNN/blob/main/Rechorus%E6%A1%86%E6%9E%B6%E6%A8%A1%E5%9E%8B%E5%88%86%E7%B1%BB.md) 按照模型应用类型和数据类型进行了简单的分类。



### 框架融合

+ 为实现HCCF融合进Rechorus，在 `src/models/general` 中增加了 [HCCF](https://github.com/lrf23/RecommendGNN/blob/main/src/models/general/HCCF.py) 模型文件。在 `src/helpers`中增加了 [HCCFRunner](https://github.com/lrf23/RecommendGNN/blob/main/src/helpers/HCCFRunner.py) 文件。
+ 模型优化：[HCCFRunnerv2](https://github.com/lrf23/RecommendGNN/blob/main/src/helpers/HCCFRunnerv2.py) 是使用了动态学习率策略的训练代码。[HCCFv2](https://github.com/lrf23/RecommendGNN/blob/main/src/models/general/HCCF_v2.py) 是调参优化但无损失函数优化的模型版本，[HCCFv3](https://github.com/lrf23/RecommendGNN/blob/main/src/models/general/HCCF_v3.py)是加上InfoNCE损失的模型版本。[HCCFv4](https://github.com/lrf23/RecommendGNN/blob/main/src/models/general/HCCF_v4.py)是针对MovieLens数据集的优化模型版本。
+ 一些测评结果在 `log` 中
+ 模型权值保存在 `model`中



### 声明
**本仓库融合了Rechorus框架和HCCF模型，仅供学习使用，部分使用到的源代码地址如下所示：**

+ [Rechorus](https://github.com/THUwangcy/ReChorus)
+ [HCCF](https://github.com/akaxlh/HCCF)



