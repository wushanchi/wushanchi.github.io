"""
配置管理器（Poor Man's Configurator）

这是一个简单的配置覆盖机制，允许通过命令行参数或配置文件来修改训练参数。

使用示例：
$ python train.py config/override_file.py --batch_size=32

执行流程：
1. 首先执行配置文件 config/override_file.py
2. 然后将 --batch_size=32 覆盖到 batch_size 变量

注意事项：
- 这不是一个 Python 模块，只是将配置代码从 train.py 中分离出来
- 配置代码直接操作 globals() 来覆盖变量
- 接受两种格式：
  1. 配置文件路径（不带 -- 前缀）
  2. 命令行参数（格式为 --key=value）
"""

import sys
from ast import literal_eval


def load_config():
    """
    遍历命令行参数并应用配置覆盖

    处理逻辑：
    1. 如果参数不包含 '='，则认为是配置文件路径，执行该文件
    2. 如果参数以 '--' 开头，则解析为 --key=value 格式的变量覆盖
    """
    for arg in sys.argv[1:]:
        if '=' not in arg:
            # 配置文件路径（不应该以 -- 开头）
            assert not arg.startswith('--')
            config_file = arg
            print(f"使用配置文件覆盖: {config_file}")
            print(f"配置文件内容:")
            with open(config_file) as f:
                print(f.read())
            exec(open(config_file).read())

        else:
            # 命令行参数格式：--key=value
            assert arg.startswith('--')
            key, val = arg.split('=')
            key = key[2:]  # 去掉前导 '--'

            if key in globals():
                try:
                    # 尝试用 literal_eval 解析值（支持 bool, int, float 等类型）
                    attempt = literal_eval(val)
                except (SyntaxError, ValueError):
                    # 如果解析失败，直接使用字符串值
                    attempt = val

                # 确保类型一致
                assert type(attempt) == type(globals()[key])

                print(f"覆盖: {key} = {attempt}")
                globals()[key] = attempt
            else:
                raise ValueError(f"未知的配置键: {key}")


# 如果作为模块导入时不执行任何操作
# 只有直接运行时才加载配置
if __name__ == '__main__':
    load_config()