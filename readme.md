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
    train: CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node 2  run_train.py --train --config_file configs/config_test.yaml
    inference: CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node 2  run_train.py --inference --config_file configs/config_test.yaml
    ```
#### 3.参数配置
- parse_args.py： 一些控制参数，比如运行模式（train/inference）
- config_test.yaml: 需要根据不同的数据集在命令行指定，包括训练参数等

  优先级：config_test.yaml > parse_args.py

