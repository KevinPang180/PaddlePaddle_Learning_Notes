# PaddlePaddle、Git & PaddleDetection安装教程(Win10环境)

## 1 PaddlePaddle安装

> 参考文档
>
> https://www.paddlepaddle.org.cn/install/quick

### 1.1 安装

打开终端(CMD或者PowerShell)，输入以下命令安装paddlepaddle，其中“-i https://mirror.baidu.com/pypi/simple”是指定从百度官方镜像源安装(推荐使用此镜像源，以确保使用的是最新的版本)

1）GPU版本

```bash
python -m pip install paddlepaddle-gpu==1.8.4.post107 -i https://mirror.baidu.com/pypi/simple
```

2）CPU版本

```bash
python -m pip install paddlepaddle==1.8.4 -i https://mirror.baidu.com/pypi/simple
```

此处需要根据你的实际情况安装对应的版本，如需安装GPU版本，需要先安装好CUDA环境，可参考我之前的视频进行安装配置。



### 1.2 验证

1）在终端中输入`python`以进入交互模式

2）在交互模式中依次输入`import paddle.fluid`和`paddle.fluid.install_check.run_check()`

如果出现`Your Paddle Fluid is installed successfully! Let's start deep learning with Paddle Fluid now`即表示安装成功



## 2 Git安装

### 2.1 下载Git安装包

https://github.com/git-for-windows/git/releases/download/v2.28.0.windows.1/Git-2.28.0-64-bit.exe

我已经下载好了，下面我就演示如何安装



### 2.2 安装

右键→以管理员方式运行→默认安装即可(当然也可根据自己的需求进行设置)

之所以要安装Git，是因为接下来要用到



## 3 PaddleDetection安装

> 参考文档
>
> https://github.com/PaddlePaddle/PaddleDetection/blob/release/0.4/docs/tutorials/INSTALL_cn.md

### 3.1 其他依赖安装

```bash
# 若Cython未安装，请安装Cython
pip install Cython

# 由于原版cocoapi不支持windows，采用第三方实现版本，该版本仅支持Python3
pip install git+https://gitee.com/KevinPang180_admin/cocoapi.git#subdirectory=PythonAPI
```

Git安装好之后，要重新打开终端，才能正常使用



Tip:

1. Linux 用户安装pycocotools时可直接用pip安装`pip install pycocotools`

2. Windows用户在安装cocoapi时需要注意，必须提前安装好Visual C ++ 2015，如您根据我之前的视频安装好了Visual Studio 2019社区版，那就可以跳过此步骤，如您的电脑上没有Visual C ++ 2015可通过此链接下载安装`https://go.microsoft.com/fwlink/?LinkId=691126`

 

### 3.2 克隆PaddleDetection

1）打开你用于克隆PaddleDetection的文件夹，进入终端并输入以下命令

我这里就直接放在下载文件夹先

```bash
git clone https://github.com/PaddlePaddle/PaddleDetection.git
```

如上述地址因网络原因无法克隆，可尝试以下地址：

```bash
git clone https://gitee.com/KevinPang180_admin/PaddleDetection.git
git clone https://gitee.com/paddlepaddle/PaddleDetection.git
```

可以看到，执行命令之后，下载目录出现了一个PaddleDetection文件夹，大小约为100M



### 3.3 安装Python依赖库

进入克隆好的PaddleDetection文件夹，在此处的终端输入`pip install -r requirements.txt`以安装依赖库

如果出现这种情况就是下载失败，直接重试即可，这样，会继续完成未下载的任务当这些依赖包都下载完成之后，会自动安装。



### 3.4 验证安装

在此终端中输入以下命令来检查安装是否成功

```bash
python ppdet/modeling/tests/test_architectures.py
```

如出现类似下面的信息，则表示安装成功

```bash
..........
----------------------------------------------------------------------
Ran 12 tests in 2.480s
OK (skipped=2)
```

 

## 4 预训练模型预测

使用预训练模型预测图像，快速体验模型预测效果：

```bash
python tools/infer.py -c configs/ppyolo/ppyolo.yml -o use_gpu=true weights=https://paddlemodels.bj.bcebos.com/object_detection/ppyolo.pdparams --infer_img=demo/000000014439_640x640.jpg
```

上述命令执行结束后，会在`output`文件夹下生成一个画有预测结果的同名图像

