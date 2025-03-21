# AI 学习规划

***

## 阶段一.打基础

目标：掌握 AI 领域核心概念、数学基础、Python 编程

### 1.数学基础

+ 线性代数（矩阵运算、特征值/向量/奇异值分解）→ 推荐课程：3Blue1Brown《线性代数》and 《线性代数应该这样学》
+ 概率论与统计(贝叶斯定理/分布函数/假设检验) → 推荐：Khan Academy 或《概率论与数理统计》
+ 微积分（梯度、偏导数、链式法则）→ 只需掌握基础求导即可 3Blue1Brown

### 2.Python & Numpy/Pandas 基础

+ Python 语法巩固 (Codecademy Python课程)
  + 构建CLI工具（argparse+logging+单元测试）
  + PyCharm配置Black/Pylint/flake8
  + 掌握virtualenv+poetry依赖管理
+ Numpy 矩阵操作
+ Matplotlib 可视化
+ Pandas 数据处理
    + 推荐：官方文档 +《effective Python》+ 《Python Data Science Handbook》
+ 自动化工具 （Jupyter Notebook + VS Code环境配置）

### 3.机器学习启蒙

+ 吴恩达《机器学习》课程（重点1-9周）
+ 掌握sklearn基础API：线性回归/K-均值聚类/决策树实战

### 4.实现项目

+ 用NumPy实现SVD分解图像压缩
+ 蒙特卡洛模拟预测硬币概率
+ 新冠疫情数据清洗与可视化仪表盘
+ 实数据集（如World Bank Data）

***

## 阶段二：机器学习实战

目标：掌握机器学习基本模型，能上手实战项目

### 1. 核心概念

+ 监督学习 :XGBoost/LightGBM实现结构化数据建模
+ 无监督学习 :DBSCAN聚类+t-SNE可视化
+ 过拟合、欠拟合、交叉验证、POC曲线、AUC值解读
+ 特征工程实战 :Kaggle Titanic完整处理流程
+ 模型解释：SHAP值分析/LIME工具使用
+ 部署入门：Flask构建简易预测API

### 2.常用模型

+ 线性回归、逻辑回归
+ 决策树、随机森林
+ SVM、KNN
+ K-Means、PCA
+ Sklearn 工具链全掌握

### 3.实战项目

+ Kaggle 上初级比赛，如 Titanic、生存预测、房价预测
+ 典型业务方向案例：用户分类、推荐系统初步、异常检测
+ 《Hands-On ML》第1-7章精读+代码复现
+ Sklearn官方示例改写（重点Pipeline/ColumnTransformer）
+ 专项训练
  + 时序特征：tsfresh库实战股票预测
  + 文本特征：TF-IDF+BERT嵌入对比
  + 自动化工具：FeatureTools应用 
  + 项目：信用卡欺诈检测系统（处理类别不平衡）
+ 模型工程化
  + 模型解释：SHAP分析贷款风控模型
  + 超参优化：Optuna对比GridSearch
  + 部署实战：FastAPI封装模型+Prometheus监控
  + 项目：完整Kaggle中级比赛（如TMDB票房预测）
***

## 阶段三：深度学习入门

目标：掌握深度学习核心架构，能训练基础神经网络

### 1. 核心知识

+ 神经网络基本结构
+ 激活函数、损失函数
+ 反向传播算法
+ 优化器（SGD, Adam）

### 2.框架学习

+ PyTorch → 推荐官方教程 + DeepLizard+《Deep Learning with PyTorch》
  + PyTorch官方Tutorial（60分钟版）
  + 从零实现MNIST分类（禁用高级API）
  + DL Fundamentals（fast.ai Part1）
+ TensorFlow (可选) 生产环境模型导出（SavedModel格式）

### 3.典型模型

+ MLP (多层感知机)
+ CNN（图像业务必备）从LeNet到ResNet实战CIFAR-10
+ RNN / LSTM（文本或时间序列业务）
+ 手写反向传播实现（推荐Andrej Karpathy的micrograd）

### 4.实战项目

+ MNIST 手写数字分类
+ 简单文本情感分类
+ 简单图像识别模型

***

## 阶段四：进阶与业务落地

目标：深度理解 AI 发展趋势，探索与业务结合

### 1. 自然语言处理（NLP）

+ 词向量 (Word2Vec, GloVe)
+ Transformer 基础 （The Annotated Transformer精读）
+ BERT, GPT 架构理解
+ Hugging Face 库掌握（Fine-tune BERT实现情感分析）

### 实战方向

+ 客服机器人
+ 文本摘要
+ 情感分析

### 2.计算机视觉（CV）

+ CNN 进阶（ResNet, EfficientNet）
+ 图像分类、目标检测、OCR
  + Albumentations实战
  + EfficientNet训练自定义数据集
  + ONNX转换+TensorRT加速
  + 项目：口罩佩戴检测系统（自定义标注数据集）

### 实战方向

+ 产品图片识别
+ 安全监控
+ OCR票据识别

### 3.推荐系统

+ 协同过滤、矩阵分解
+ DeepFM, Wide & Deep 模型
+ 公司用户数据可应用推荐算法

***

## 阶段五：大模型、AIGC & 未来趋势（持续跟进）

目标：了解 AIGC（生成式 AI）、大语言模型(LLM)、部署与优化

### 1. 大模型基础

+ Transformer 深入
+ GPT 系列、LLAMA、Claude 基本原理
+ Fine-tuning、Prompt Engineering
+ LoRA、模型蒸馏等轻量化方法

### 2.AIGC 方向

+ 图像生成 (Stable Diffusion, DALL·E)
+ 文本生成 (ChatGPT, GPT-4/5 API 调用)
+ 视频、音频生成趋势了解

### 3.部署与工程化

+ ONNX、TorchScript 模型部署
+ API 接口封装
+ GPU 服务器配置、云平台使用 (AWS/GCP)

***

## 资源推荐

| 阶段            | 资源推荐                                             |
|:--------------|:-------------------------------------------------|
| 数学基础	         | 	3Blue1Brown YouTube, Khan Academy               |
| Python/Numpy	 | 官方文档 +《Python 数据科学手册》                            |
| 机器学习	         | Andrew Ng《机器学习》+ Kaggle                          |
| 深度学习	         | DeepLizard, PyTorch 官方教程                         |
| NLP	          | Hugging Face 官方文档, 《The Illustrated Transformer》 |
| CV            | FastAI 课程, Stanford CS231n                       |
| 推荐系统	         | Coursera 推荐系统专栏                                  |
| 大模型	          | OpenAI 文档, Papers with Code, Hugging Face Spaces |

注：需要有笔记代码输出，找真是数据集动手  
关注：Twitter, arXiv, Papers with Code  
社区：Kaggle、Hugging Face 社区
***