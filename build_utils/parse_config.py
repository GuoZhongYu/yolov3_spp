import os
import numpy as np

'''
函数的目的在于，将.cfg中的内容提取出来，并且进行划分，简单来说就是解析.cfg文件的内容
形成对应的字典或者是列表，为后续的网络搭建做准备
'''
def parse_model_cfg(path: str):
    # 传入.cfg文件路径，检查文件是否存在
    if not path.endswith(".cfg") or not os.path.exists(path):
        raise FileNotFoundError("the cfg file not exist...")

    # 读取文件信息，并通过换行符进行划分
    with open(path, "r") as f:
        lines = f.read().split("\n")

    # 将有数据的并且不是#开头的行保留下来，即去除空行和注释行
    lines = [x for x in lines if x and not x.startswith("#")]
    # 去除每行开头和结尾的空格符，strip()去除空格符号
    lines = [x.strip() for x in lines]

    mdefs = []  # module definitions
    # 读取除去注释、空行、空格符后的每一行
    for line in lines:
        # 如果行是以[开头，则说明这是一个网络层，包含多个数据
        if line.startswith("["):  # this marks the start of a new block
            # 往mdefs中加入一个字典
            mdefs.append({})
            # 通过[-1]从最后开始数，找到加入的字典，将type置为[XXXXX]
            mdefs[-1]["type"] = line[1:-1].strip()  # 记录module类型
            # 如果是卷积模块，设置默认不使用BN(普通卷积层后面会重写成1，最后的预测层conv保持为0)
            # 这里的意思是，先将所有的卷积层的BN都设置为0，也就是默认为0，哪些conv需要BN，会出现batch_normalize=1，出现的时候再将其置为1
            if mdefs[-1]["type"] == "convolutional":
                mdefs[-1]["batch_normalize"] = 0
        else:
            # 只有[XXXX]需要特殊的操作，其他的只需要通过=将其划分为两部分即可
            key, val = line.split("=")
            key = key.strip()
            val = val.strip()

            # 如果key是anchors
            if key == "anchors":
                # anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
                val = val.replace(" ", "")  # 将空格去除
                # 将anchors的数据转换为浮点数并且两个一组
                '''
                array([[ 10.,  13.],
                       [ 16.,  30.],
                       [ 33.,  23.],
                       [ 30.,  61.],
                       [ 62.,  45.],
                       [ 59., 119.],
                       [116.,  90.],
                       [156., 198.],
                       [373., 326.]])
                '''
                mdefs[-1][key] = np.array([float(x) for x in val.split(",")]).reshape((-1, 2))  # np anchors
            # 如果key是["from", "layers", "mask"]里面的值，或者key满足key为size并且有逗号（这里的包含逗号，指的是有的size是按照3，3来写的）
            elif (key in ["from", "layers", "mask"]) or (key == "size" and "," in val):
                mdefs[-1][key] = [int(x) for x in val.split(",")]
            else:
                # TODO: .isnumeric() actually fails to get the float case
                if val.isnumeric():  # return int or float 如果是数值的情况
                    mdefs[-1][key] = int(val) if (int(val) - float(val)) == 0 else float(val)
                else:
                    mdefs[-1][key] = val  # return string  是字符的情况

    # check all fields are supported
    supported = ['type', 'batch_normalize', 'filters', 'size', 'stride', 'pad', 'activation', 'layers', 'groups',
                 'from', 'mask', 'anchors', 'classes', 'num', 'jitter', 'ignore_thresh', 'truth_thresh', 'random',
                 'stride_x', 'stride_y', 'weights_type', 'weights_normalization', 'scale_x_y', 'beta_nms', 'nms_kind',
                 'iou_loss', 'iou_normalizer', 'cls_normalizer', 'iou_thresh', 'probability', 'flag_SEB']

    # 遍历检查每个模型的配置
    for x in mdefs[1:]:  # 0对应net配置
        # 遍历每个配置字典中的key值
        for k in x:
            if k not in supported:
                raise ValueError("Unsupported fields:{} in cfg".format(k))
    # 返回的是一个元素为字典的列表
    return mdefs

# 输入所在的路径，读取路径文件的内容，并返回字典类型
def parse_data_cfg(path):
    # Parses the data configuration file
    if not os.path.exists(path) and os.path.exists('data' + os.sep + path):  # add data/ prefix if omitted
        path = 'data' + os.sep + path

    with open(path, 'r') as f:
        lines = f.readlines()

    options = dict()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, val = line.split('=')
        options[key.strip()] = val.strip()

    return options
