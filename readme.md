### 自己写的使用Pytorch进行模型训练的基本框架（以mnist为例），支持单卡多GPU
#### 1.目录结构
```shell
├── configs
│   └── config_test.yaml  # 配置文件
├── data  # 数据集
│   └── MNIST
├── logs # 日志，可以选择是否写入到文件
│   ├── 2022-04-20-14-13.log
├── parse_args.py # 参数解析文件
├── pre_models # 存放预训练模型
├── readme.md 
├── result  # 结果文件
│   └── model_save
│       └── mnist
│           ├── Net-epoch-0.pkl
│           ├── Net-epoch-1.pkl
│           └── Net-model_best_dict.pkl
├── run_train.py # 主程序入口文件
├── src
│   └── models # 自定义模型
│       ├── mnist_net.py
└── utils
    ├── data_utils.py # 数据工具
    ├── distributed_utils.py # 分布式训练工具
    ├── logger.py # 日志配置
    └── yaml_utils.py # yaml参数读取
```

#### 2.启动方式
- 命令行启动：使用pytorch提供的torchrun进行启动，具体信息可查看[torchrun](https://pytorch.org/docs/stable/elastic/run.html#module-torch.distributed.run)
    ```shell
    CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node 2  run_train.py --train --config_file configs/config_test.yaml  # train
    CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node 2  run_train.py --inference --config_file configs/config_test.yaml  # inference
    ```
- 如果使用单GPU进行训练，可以将可见GPU设置为想使用的GPU编号，nproc_per_node设置为1，或者不设置（默认为可见GPU的第一个）
    ```shell
    CUDA_VISIBLE_DEVICES=1 torchrun  run_train.py --train --config_file configs/config_test.yaml # train
    CUDA_VISIBLE_DEVICES=1 torchrun  run_train.py --inference --config_file configs/config_test.yaml # inference
    ```
#### 3.参数配置
- parse_args.py： 一些控制参数，比如运行模式（train/inference）
- config_test.yaml: 需要根据不同的数据集在命令行指定，包括训练参数等

  优先级：config_test.yaml > parse_args.py

  注意：总的batch_size为设置的batch_size*gpu_nums

#### 4. 动态学习率 

 实现了两种方式的学习率调整策略：
+ warm_up + 余弦退火
+ warm_up + stepRL
 
有一个bug：多GPU训练时，如果中途停止训练，继续进行训练时，开始学习率会变为之前训练的最后一个epoch的学习率，之后恢复正常
。比如：总共训练epoches=10个，当epoch=5时训练停止，继续训练时，epoch=6的学习率与epoch=5的学习率相同
  （实在不会改了，估计影响不会太大）