## TensorFlow简介
[TensorFlow](https://www.tensorflow.org/)是一个端到端的开源平台用于机器学习。它有一个全面、灵活的生态系统[工具](https://www.tensorflow.org/resources/tools)，[库](https://www.tensorflow.org/resources/libraries-extensions)和[社区](https://www.tensorflow.org/community)资源让研究人员推动ML的最新技术，开发人员轻松构建和部署ML驱动的应用程序。TensorFlow最初由研究人员和工程师开发谷歌机器智能研究组织中的谷歌大脑团队进行机器学习和深度神经网络研究。系统是足够通用以至于能够应用于其他各种各样的领域。这些应用得意于DAS软件栈对Tensorflow 常用算子及网络模型的支持，开发者针对DCU加速卡开发应用时，可以便捷地调用深度学习以及各类数据科学应用开发所需的算子，灵活地构造各类深度神经网络模型以及其他机器学习领域的算法。

## 版本约束 

#### 暂不支持的官方版本或功能

更高版本：暂不支持tensorflow2.14以上

#### TensorFlow软件版本配套关系

tensorflow版本：2.13.1

DCU适配版tensorflow软件包版本:2.13.1+das.opt1.dtk24042

DTK版本:21.04.1

## 前置条件

使用 DAS PyTorch需要参考[《DCU新手入门教程》](https://developer.hpccube.com/gitbook/dcu_tutorial/index.html)在主机系统安装以下组件:
- DCU 驱动程序
- DTK
- Docker引擎

## 使用命令安装tensorflow

工具安装使用 pip 方式，从光源社区[DAS](https://cancon.hpccube.com:65024/4/main/)中下载此工具的安装包。注意与 tensorflow 版本匹配
```
pip install tensorflow* (下载的tensorflow的whl包)
```

## 验证

安装完成之后，可通过以下指令验证是否安装成功,指令执行后会显示当前tensorflow的版本号。
```
python -c "import tensorflow; print(tensorflow.__version__)"
```
## 建议阅读


有关Tensorflow的更多信息，请参见:
- [AI生态包](https://cancon.hpccube.com:65024/4/main/)
- [Model Zoo](https://sourcefind.cn/#/model-zoo/list)
- [DCU Toolkit](https://cancon.hpccube.com:65024/1/main)
- [驱动](https://cancon.hpccube.com:65024/6/main)
- [学习中心](https://developer.hpccube.com/study/)
- [论坛](https://forum.hpccube.com/)



