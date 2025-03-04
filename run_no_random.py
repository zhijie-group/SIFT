import os
import numpy as np
import torch
from opencompass.cli.main import main


def set_random_seed():
    # 获取环境变量中的随机种子
    random_seed = int(os.environ.get('RANDOM_SEED', 42))
    # 设置 numpy 的随机数生成器种子
    np.random.seed(random_seed)
    # 设置 torch 的随机数生成器种子
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed) # 为所有的 GPU 设置随机数生成器种子
        torch.backends.cudnn.deterministic = True # 为了保证实验结果的可重复性，需要关闭 cudnn 的随机性
        torch.backends.cudnn.benchmark = False # 为了保证实验结果的可重复性，需要关闭 benchmark



if __name__ == '__main__':
    set_random_seed()
    main()