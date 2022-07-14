from build_utils.layers import *
from build_utils.parse_config import *

ONNX_EXPORT = False


def create_modules(modules_defs: list, img_size):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    :param modules_defs: 通过.cfg文件解析得到的每个层结构的列表
    :param img_size:
    :return:
    """
    # isinstance() 函数来判断一个对象是否是一个已知的类型，将数值img_size变为[img_size,img_size]形式
    img_size = [img_size] * 2 if isinstance(img_size, int) else img_size
    # 删除解析cfg列表中的第一个配置(对应[net]的配置)
    modules_defs.pop(0)  # cfg training hyperparams (unused)

    output_filters = [3]  # input channels
    module_list = nn.ModuleList()
    # 统计哪些特征层的输出会被后续的层使用到(可能是特征融合，也可能是拼接)
    # 在残差模块、spp模块、拼接模块会使用
    routs = []  # list of layers which rout to deeper layers
    yolo_index = -1

    # 遍历搭建每个层结构，enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。
    # 这里的i代表着下标，是从0开始的，代表着计算机中的索引值
    for i, mdef in enumerate(modules_defs):
        modules = nn.Sequential()

        # 如果是带有BN的卷积层
        # 其实是可以看到，这里的卷积层、BN层和激活层是依次加入的，只能说组成的convolutional是一个模块，而不能说convolutional是一个整体
        # 上面说到的模块和整体，模块可以理解拥有多个功能，每个功能对应一个函数，以此来共同组成一个功能。而整体理解为，虽然有多个功能，但是这些功能都在一个函数里面完成
        if mdef["type"] == "convolutional":
            # 查看是否是使用BN层
            bn = mdef["batch_normalize"]  # 1 or 0 / use or not
            # 卷积核个数，也代表着输出维度的多少
            filters = mdef["filters"]
            k = mdef["size"]  # kernel size
            # 如果stride是两个值，则赋值为（x，y），只有一个值就只赋值为1个值
            stride = mdef["stride"] if "stride" in mdef else (mdef['stride_y'], mdef["stride_x"])
            # 如果k = mdef["size"]，也就是卷积核的数值是整数，则创建这个层的基本信息
            if isinstance(k, int):
                # 几个注意的点，输出维度数=卷积核的个数，padding的计算方式，bias偏置和是否有bn层有关
                modules.add_module("Conv2d", nn.Conv2d(in_channels=output_filters[-1],
                                                       out_channels=filters,
                                                       kernel_size=k,
                                                       stride=stride,
                                                       padding=k // 2 if mdef["pad"] else 0,
                                                       bias=not bn))
            else:
                raise TypeError("conv2d filter size must be int type.")
            # 如果使用bn，则在网络中加入bn层，输入的参数为输入图像的通道数量，bn是在卷积之后才有的，卷积完成之后通道数变为卷积和个数，所以参数为卷积核个数
            if bn:
                modules.add_module("BatchNorm2d", nn.BatchNorm2d(filters))
            else:
                # 在名字为convolutional的前提下，包含两种不同的卷积层，可以通过是否有bn层来进行划分
                # 如果该卷积操作没有bn层，意味着该层为yolo的predictor
                # routs的作用是什么？ 这里的意思是将yolo前面的predictor的索引值记录下来
                routs.append(i)  # detection output (goes into yolo layer)

            if mdef["activation"] == "leaky":
                modules.add_module("activation", nn.LeakyReLU(0.1, inplace=True))
            else:
                pass

        elif mdef["type"] == "BatchNorm2d":
            pass

        elif mdef["type"] == "maxpool":
            k = mdef["size"]  # kernel size
            stride = mdef["stride"]
            modules = nn.MaxPool2d(kernel_size=k, stride=stride, padding=(k - 1) // 2)

        elif mdef["type"] == "upsample":
            if ONNX_EXPORT:  # explicitly state size, avoid scale_factor
                g = (yolo_index + 1) * 2 / 32  # gain
                modules = nn.Upsample(size=tuple(int(x * g) for x in img_size))
            else:
                modules = nn.Upsample(scale_factor=mdef["stride"])

        # 如果是route层，route层如果只有一层那么就只是拿过来用，如果是多个层，那么就是将多层的数据在通道层面连接起来
        elif mdef["type"] == "route":  # [-2],  [-1,-3,-5,-6], [-1, 61]
            layers = mdef["layers"]
            # 通道数的计算，计算所有的通道数的总和，sum（）
            filters = sum([output_filters[l + 1 if l > 0 else l] for l in layers])
            # list.extend() 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）
            # 注意这里extend的是 i ，也就是用到的层的索引值
            routs.extend([i + l if l < 0 else l for l in layers])
            # Concat层的作用就是将两个及以上的特征图在channel上进行连接
            modules = FeatureConcat(layers=layers)

        # 如果是shortcut，shortcut层的作用是将两层进行连接，from代表该层和前面的那一层进行连接
        elif mdef["type"] == "shortcut":
            layers = mdef["from"]
            filters = output_filters[-1]
            # routs.extend([i + l if l < 0 else l for l in layers])
            # 注意这里extend的是 i ，也就是用到的层的索引值
            routs.append(i + layers[0])
            # 加权特征融合
            modules = WeightedFeatureFusion(layers=layers, weight="weights_type" in mdef)

        # 如果是yolo层，yolo层的mask代表的是选取哪几个值
        elif mdef["type"] == "yolo":
            yolo_index += 1  # 记录是第几个yolo_layer [0, 1, 2]
            stride = [32, 16, 8]  # 预测特征层对应原图的缩放比例

            modules = YOLOLayer(anchors=mdef["anchors"][mdef["mask"]],  # anchor list
                                nc=mdef["classes"],  # number of classes
                                img_size=img_size,
                                stride=stride[yolo_index])

            # Initialize preceding Conv2d() bias (https://arxiv.org/pdf/1708.02002.pdf section 3.3)
            # 对yolo前一层的卷积层的偏置进行初始化，在前面的设置中，有bn层的卷积是没有偏置的，没有bn层的也就是yolo前面的卷积层是存在偏置的，所以就需要将偏置进行初始化
            try:
                j = -1
                # bias: shape(255,) 索引0对应Sequential中的Conv2d
                # view: shape(3, 85)
                b = module_list[j][0].bias.view(modules.na, -1)
                b.data[:, 4] += -4.5  # obj
                b.data[:, 5:] += math.log(0.6 / (modules.nc - 0.99))  # cls (sigmoid(p) = 1/nc)
                module_list[j][0].bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            except Exception as e:
                print('WARNING: smart bias initialization failure.', e)
        else:
            print("Warning: Unrecognized Layer Type: " + mdef["type"])

        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    # routs在三个地方进行了修改，记录了yolo的predictor的索引，route、shortcut所用到的层的索引值
    # 创建一个list，里面的值全为False，当routs有值的时候，或者说是会被用到的时候，就将其置为True
    routs_binary = [False] * len(modules_defs)
    for i in routs:
        routs_binary[i] = True
    return module_list, routs_binary


class YOLOLayer(nn.Module):
    """
    对YOLO的输出进行处理
    """
    def __init__(self, anchors, nc, img_size, stride):
        super(YOLOLayer, self).__init__()
        self.anchors = torch.Tensor(anchors)
        self.stride = stride  # layer stride 特征图上一步对应原图上的步距 [32, 16, 8]
        self.na = len(anchors)  # number of anchors (3)
        self.nc = nc  # number of classes (80)
        self.no = nc + 5  # number of outputs (85: x, y, w, h, obj, cls1, ...)
        # nx，ny代表所使用的预测特征层的高度和宽度，ng代表cell的大小，这里是初始化操作，后续的正向传播过程中会对其进行进一步的赋值
        self.nx, self.ny, self.ng = 0, 0, (0, 0)  # initialize number of x, y gridpoints
        # 将anchors大小缩放到grid尺度，之前的anchor都是针对原图大小的，这里就是通过除以stride操作来实现缩放到对应的特征图大小上
        # anchor_vec应该是3个2维的向量组成的tensor，对应着有3个anchor，每个anchor包含两个参数
        self.anchor_vec = self.anchors / self.stride
        # batch_size, na, grid_h, grid_w, wh（anchor的宽度和高度），这里的view()操作作用等同于reshape操作，对大小进行一个重设
        # 值为1的维度对应的值不是固定值，后续操作可根据broadcast广播机制自动扩充
        self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2)
        self.grid = None

        if ONNX_EXPORT:
            self.training = False
            self.create_grids((img_size[1] // stride, img_size[0] // stride))  # number x, y grid points

    def create_grids(self, ng=(13, 13), device="cpu"):
        """
        更新grids信息并生成新的grids参数
        :param ng: 特征图大小
        :param device:
        :return:
        """
        self.nx, self.ny = ng
        self.ng = torch.tensor(ng, dtype=torch.float)

        # build xy offsets 构建每个cell处的anchor的xy偏移量(在feature map上的)
        if not self.training:  # 训练模式不需要回归到最终预测boxes
            yv, xv = torch.meshgrid([torch.arange(self.ny, device=device),
                                     torch.arange(self.nx, device=device)])
            # batch_size, na, grid_h, grid_w, wh
            self.grid = torch.stack((xv, yv), 2).view((1, 1, self.ny, self.nx, 2)).float()

        if self.anchor_vec.device != device:
            self.anchor_vec = self.anchor_vec.to(device)
            self.anchor_wh = self.anchor_wh.to(device)

    # p代表的是预测值
    def forward(self, p):
        if ONNX_EXPORT:
            bs = 1  # batch size
        else:
            bs, _, ny, nx = p.shape  # batch_size, predict_param(255), grid(13), grid(13)
            # 第一次训练或者说是cell大小不一样，则需要通过create_grids重新设置
            if (self.nx, self.ny) != (nx, ny) or self.grid is None:  # fix no grid bug
                self.create_grids((nx, ny), p.device)

        # view: (batch_size, 255, 13, 13) -> (batch_size, 3, 85, 13, 13)
        # permute: (batch_size, 3, 85, 13, 13) -> (batch_size, 3, 13, 13, 85)
        # [bs, anchor, grid, grid, xywh + obj + classes]
        p = p.view(bs, self.na, self.no, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        if self.training:
            return p
        elif ONNX_EXPORT:
            # Avoid broadcasting for ANE operations
            m = self.na * self.nx * self.ny  # 3*
            ng = 1. / self.ng.repeat(m, 1)
            grid = self.grid.repeat(1, self.na, 1, 1, 1).view(m, 2)
            anchor_wh = self.anchor_wh.repeat(1, 1, self.nx, self.ny, 1).view(m, 2) * ng

            p = p.view(m, self.no)
            # xy = torch.sigmoid(p[:, 0:2]) + grid  # x, y
            # wh = torch.exp(p[:, 2:4]) * anchor_wh  # width, height
            # p_cls = torch.sigmoid(p[:, 4:5]) if self.nc == 1 else \
            #     torch.sigmoid(p[:, 5:self.no]) * torch.sigmoid(p[:, 4:5])  # conf
            p[:, :2] = (torch.sigmoid(p[:, 0:2]) + grid) * ng  # x, y
            p[:, 2:4] = torch.exp(p[:, 2:4]) * anchor_wh  # width, height
            p[:, 4:] = torch.sigmoid(p[:, 4:])
            p[:, 5:] = p[:, 5:self.no] * p[:, 4:5]
            return p
        else:  # inference
            # [bs, anchor, grid, grid, xywh + obj + classes]
            io = p.clone()  # inference output
            io[..., :2] = torch.sigmoid(io[..., :2]) + self.grid  # xy 计算在feature map上的xy坐标
            io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh  # wh yolo method 计算在feature map上的wh
            io[..., :4] *= self.stride  # 换算映射回原图尺度
            torch.sigmoid_(io[..., 4:])
            return io.view(bs, -1, self.no), p  # view [1, 3, 13, 13, 85] as [1, 507, 85]


class Darknet(nn.Module):
    """
    YOLOv3 spp object detection model

    verbose是否打印每个模块的详细信息，默认是不打印
    """
    def __init__(self, cfg, img_size=(416, 416), verbose=False):
        super(Darknet, self).__init__()
        # 这里传入的img_size只在导出ONNX模型时起作用
        self.input_size = [img_size] * 2 if isinstance(img_size, int) else img_size
        # 解析网络对应的.cfg文件，通过parse_model_cfg函数解析文件的内容，通过parse_model_cfg(cfg)返回的是一个元素为字典的列表
        self.module_defs = parse_model_cfg(cfg)
        # 根据解析的网络结构一层一层去搭建，返回的是module_list, routs_binary（list，元素为bool，有需要的值为True）
        self.module_list, self.routs = create_modules(self.module_defs, img_size)
        # 获取所有YOLOLayer层的索引，返回的是层名为yolo的索引
        self.yolo_layers = get_yolo_layers(self)

        # 打印下模型的信息，如果verbose为True则打印详细信息
        self.info(verbose) if not ONNX_EXPORT else None  # print model description

    def forward(self, x, verbose=False):
        return self.forward_once(x, verbose=verbose)

    def forward_once(self, x, verbose=False):
        # yolo_out收集每个yolo_layer层的输出
        # out收集每个模块的输出
        yolo_out, out = [], []
        if verbose:
            print('0', x.shape)
            str = ""

        for i, module in enumerate(self.module_list):
            name = module.__class__.__name__
            # 如果是属于shortcut或者是toute层
            if name in ["WeightedFeatureFusion", "FeatureConcat"]:  # sum, concat
                if verbose:
                    l = [i - 1] + module.layers  # layers
                    sh = [list(x.shape)] + [list(out[i].shape) for i in module.layers]  # shapes
                    str = ' >> ' + ' + '.join(['layer %g %s' % x for x in zip(l, sh)])
                # module(x)或者module(x,out)简单理解为计算？
                x = module(x, out)  # WeightedFeatureFusion(), FeatureConcat()
            # 如果是YOLOLayer层，就将结果放入到yolo_out中
            elif name == "YOLOLayer":
                yolo_out.append(module(x))
            else:  # run module directly, i.e. mtype = 'convolutional', 'upsample', 'maxpool', 'batchnorm2d' etc.
                x = module(x)

            # 如果routs的某一个为True，则放入计算后的结果；如果为False，则放入[]
            # 将yolo的predictor、shortcut和route层需要用到的层的数据放入到out中
            out.append(x if self.routs[i] else [])
            if verbose:
                print('%g/%g %s -' % (i, len(self.module_list), name), list(x.shape), str)
                str = ''

        # 如果是在训练模式，则返回yolo_out的结果
        if self.training:  # train
            return yolo_out
        elif ONNX_EXPORT:  # export
            # x = [torch.cat(x, 0) for x in zip(*yolo_out)]
            # return x[0], torch.cat(x[1:3], 1)  # scores, boxes: 3780x80, 3780x4
            p = torch.cat(yolo_out, dim=0)

            # # 根据objectness虑除低概率目标
            # mask = torch.nonzero(torch.gt(p[:, 4], 0.1), as_tuple=False).squeeze(1)
            # # onnx不支持超过一维的索引（pytorch太灵活了）
            # # p = p[mask]
            # p = torch.index_select(p, dim=0, index=mask)
            #
            # # 虑除小面积目标，w > 2 and h > 2 pixel
            # # ONNX暂不支持bitwise_and和all操作
            # mask_s = torch.gt(p[:, 2], 2./self.input_size[0]) & torch.gt(p[:, 3], 2./self.input_size[1])
            # mask_s = torch.nonzero(mask_s, as_tuple=False).squeeze(1)
            # p = torch.index_select(p, dim=0, index=mask_s)  # width-height 虑除小目标
            #
            # if mask_s.numel() == 0:
            #     return torch.empty([0, 85])

            return p
        # 如果不是在训练模式
        else:  # inference or test
            x, p = zip(*yolo_out)  # inference output, training output
            x = torch.cat(x, 1)  # cat yolo outputs

            return x, p

    def info(self, verbose=False):
        """
        打印模型的信息
        :param verbose:
        :return:
        """
        torch_utils.model_info(self, verbose)


def get_yolo_layers(self):
    """
    获取网络中三个"YOLOLayer"模块对应的索引
    :param self:
    :return:
    """
    return [i for i, m in enumerate(self.module_list) if m.__class__.__name__ == 'YOLOLayer']  # [89, 101, 113]



