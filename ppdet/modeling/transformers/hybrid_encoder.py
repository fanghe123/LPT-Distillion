# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register, serializable
from ppdet.modeling.ops import get_act_fn
from ..shape_spec import ShapeSpec
from ..backbones.csp_darknet import BaseConv
from ..backbones.cspresnet import RepVggBlock
from ppdet.modeling.transformers.detr_transformer import TransformerEncoder
from ..initializer import xavier_uniform_, linear_init_
from ..layers import MultiHeadAttention
from paddle import ParamAttr
from paddle.regularizer import L2Decay
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import os
__all__ = ['HybridEncoder']

__all__ = ['HybridEncoder']
class AMSELayer(nn.Layer):
    def __init__(self,channel,reduction=4):
        super(AMSELayer,self).__init__()
        "全局平均池化和全局最大池化组成的amse"
        self.gap = nn.AdaptiveAvgPool2D(1)
        #self.gmp = nn.AdaptiveAvgPool2d(1)
        # self.linar_a = nn.Sequential(
        #     nn.Conv2d(channel,channel//4,1,bias=False),
        #     nn.BatchNorm2d(channel//4),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(channel//4, 16 , 1, bias=False),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(inplace=True),
        # )
        # self.linar_m = nn.Sequential(
        #     nn.Conv2d(channel, channel // 4, 1, bias=False),
        #     nn.BatchNorm2d(channel // 4),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(channel // 4, 16, 1, bias=False),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(inplace=True),
        #
        self.fc = nn.Sequential(
        nn.Linear(channel, channel // reduction, bias_attr=False),
        nn.ReLU(),
        nn.Linear(channel // reduction, channel, bias_attr=False)
        )
        self.active = nn.Sigmoid()

    def forward(self,x):
        b, c, _, _ = x.shape
        x_a = self.gap(x)  #将输入的特征图进行平均池化，然后输出一个大小为 1x1 的特征图

        y = paddle.matmul(paddle.reshape(x_a, [b, c, 1]), paddle.reshape(x_a, [b, 1, c])) # b c  c
        # b c c ---> b c 1 1

        y = paddle.sum(y,axis=1) #最终的输出张量 y 的维度为 (b, c)，其中 b 表示批次大小，c 表示样式特征向量的长度。

        y = self.fc(y)

        y = self.active((paddle.reshape(y, [b, c, 1, 1]) + x_a))
        return x * paddle.expand_as(y, x)
class PF(nn.Layer):
    def __init__(self,in_channels=256,out_channels=64):
        super(PF,self).__init__()
        self.head1 = Head3(in_channels, in_channels//2, reduction=4, h=64)  # 这个数等于输入图片数/8
        self.head2 = Head4(in_channels//2, out_channels, reduction=4, h=64, softmax=True)


#通过证明 其原始代码放进来是可以行的通的 现在开始修改代码
    def forward(self,x):
        x_ = self.head1(x)
        pfx_qk = self.head2(x_)
        return pfx_qk

class CF(nn.Layer):
    def __init__(self,in_channels=256,out_channels=64):
        super(CF,self).__init__()
        self.psp_a = _PSPModule(in_channels,out_channels)
        self.psp_m = _PSPModule(in_channels,out_channels) #,model='max'
        self.qk = Pool_transform(out_channels)


    def forward(self, x):
        x_pool_a = self.psp_a(x)
        x_pool_m = self.psp_m(x)
        x_qk = self.qk(x_pool_a, x_pool_m)

        return x_qk


class _PSPModule(nn.Layer):
    def __init__(self, in_channels,out_channels, bin_sizes=[1, 2, 3, 6],model='avg'):
        super(_PSPModule, self).__init__()
        self.model = model
        self.out_chanels = out_channels
        "池化并调整通道"
        self.stages = nn.LayerList([self._make_stages(in_channels, out_channels, b_s)
                                      for b_s in bin_sizes])

    def _make_stages(self, in_channels, out_channels, bin_sz):
        if self.model =='avg':
            prior = nn.AdaptiveAvgPool2D(output_size=bin_sz)
        else:
            prior = nn.AdaptiveMaxPool2D(output_size=bin_sz)
        conv = nn.Conv2D(in_channels, out_channels, kernel_size=1, bias_attr=False)
        bn = nn.BatchNorm2D(out_channels)
        relu = nn.ReLU()
        return nn.Sequential(prior, conv, bn, relu)

    def forward(self, features):
        b,c,h, w = features.shape
        pyramids = []
        for stage in self.stages:
            pyramid = stage(features)
            pyramid = pyramid.reshape([b, self.out_chanels, -1])
            pyramids.append(pyramid)
        out = paddle.concat(pyramids,axis=2)

        return out

class Pool_transform(nn.Layer):
    def __init__(self,in_channels):
        super(Pool_transform,self).__init__()
        self.linear_transform1 = nn.Sequential(nn.Conv2D(in_channels, in_channels//4, 1, bias_attr=False),
                     nn.BatchNorm2D(in_channels//4), nn.ReLU(),
                     nn.Conv2D(in_channels//4, in_channels, 1,bias_attr=False),
                 nn.BatchNorm2D(in_channels ), nn.ReLU()
                      )
        self.linear_transform2 = nn.Sequential(nn.Conv2D(in_channels, in_channels // 4, 1, bias_attr=False),
                                               nn.BatchNorm2D(in_channels // 4), nn.ReLU(),
                                               nn.Conv2D(in_channels // 4, in_channels, 1, bias_attr=False),
                                               nn.BatchNorm2D(in_channels), nn.ReLU()
                                               )

    def forward(self,pool_a,pool_m):
        b,c,n = pool_a.shape
        #增加维度 4个维度才能使用卷积
        x_q = self.linear_transform1(pool_a[..., None])  # b c n 1
        x_k = self.linear_transform2(pool_m[..., None]).reshape([b, c, n]).transpose([0, 2, 1])  # b n c
        out = paddle.bmm(x_q.reshape([b, c, n]), x_k)  # b c c
        return out
class Head3(nn.Layer):
    def __init__(self, in_channels, out_channels, reduction=4, h=64,softmax=False):
        super(Head3, self).__init__()
        self.h = h
        self.sof = softmax            #此处使用了3*3的卷积核仅仅用于降低维度  1*1的卷积核也可以降低维度
        self.query =  nn.Sequential(nn.Conv2D(in_channels, out_channels, 3,padding=1,bias_attr=False),nn.BatchNorm2D(out_channels),nn.ReLU())
        self.key =  nn.Sequential(nn.Conv2D(in_channels, out_channels, 3,padding=1,bias_attr=False),nn.BatchNorm2D(out_channels),nn.ReLU())

        self.pool_h = nn.AdaptiveMaxPool2D((1, None))
        self.pool_w = nn.AdaptiveMaxPool2D((None, 1))



        self.softmax = nn.Softmax(axis=-1)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)

        b, c, h, w = query.shape
        #print(b, c, h, w)
        q_h = self.pool_h(query)
        k_w = self.pool_w(key)

        q_h = q_h.reshape([b * c, w, -1])
        k_w = k_w.reshape([b * c, -1, h])

        energy = paddle.bmm(q_h, k_w).reshape([b, c, h, w])  # b c h w
        if self.sof:
            attention = F.softmax(energy)
            out = attention
        else:
            # out = torch.mul(attention, value)                   .view(b, c, h, w)
            # out = self.gamma * out + x
            out = energy
        return out
class Head4(nn.Layer):
    def __init__(self, in_channels, out_channels, reduction=4, h=64,softmax=False):
        super(Head4, self).__init__()
        self.h = h
        self.sof = softmax            #此处使用了3*3的卷积核仅仅用于降低维度  1*1的卷积核也可以降低维度
        self.query =  nn.Sequential(nn.Conv2D(in_channels, out_channels, 3,padding=1,bias_attr=False),nn.BatchNorm2D(out_channels),nn.ReLU())
        self.key =  nn.Sequential(nn.Conv2D(in_channels, out_channels, 3,padding=1,bias_attr=False),nn.BatchNorm2D(out_channels),nn.ReLU())

        self.pool_h = nn.AdaptiveAvgPool2D((1, None))
        self.pool_w = nn.AdaptiveAvgPool2D((None, 1))


        self.softmax = nn.Softmax(axis=-1)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)

        b, c, h, w = query.shape
        #print(b, c, h, w)
        q_h = self.pool_h(query)
        k_w = self.pool_w(key)

        q_h = q_h.reshape([b * c, w, -1])
        k_w = k_w.reshape([b * c, -1, h])

        energy = paddle.bmm(q_h, k_w).reshape([b, c, h, w])  # b c h w
        if self.sof:
            attention = F.softmax(energy)
            out = attention
        else:
            # out = torch.mul(attention, value)                   .view(b, c, h, w)
            # out = self.gamma * out + x
            out = energy
        return out

class MyAttention(nn.Layer):
    def __init__(self,in_channels=256,out_channels=64):
        super(MyAttention, self).__init__()
        norm_layer = nn.BatchNorm2D
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.amse1 = AMSELayer(256)
        self.pf = PF(256,64)
        self.cf = CF(256,64)
        self.after_bacbone_p = nn.Sequential(
            nn.Conv2D(256, 256, 3, padding=1,bias_attr=False),
            norm_layer(256),
            nn.ReLU())
        self.value = nn.Sequential(nn.Conv2D(in_channels, out_channels, 3, padding=1, bias_attr=False),
                                   nn.BatchNorm2D(out_channels),
                                   nn.ReLU(), )
        self.reduce = nn.Sequential(nn.Conv2D(in_channels, out_channels, 3, padding=1, bias_attr=False),
                                    nn.BatchNorm2D(out_channels),
                                    nn.ReLU())
        self.huanyuan = nn.Conv2D(out_channels, 256, kernel_size=3,
                              stride=1, padding=1)

        self.gamma =paddle.create_parameter(shape=[1], dtype='float32')
    def forward(self, x):
        x1 = self.amse1(x)
        x1 = self.after_bacbone_p(x1)
        x_p = self.pf(x1)
        x_c = self.cf(x1)
        x_value = self.value(x)
        b, c, h, w = x_value.shape
        x_att1 = paddle.multiply(x_p, x_value)
        x_inner = paddle.bmm(x_c, x_value.reshape([b,-1,h*w]))
        x_att2 = x_inner.reshape([b, c, h, w])
        x_att2 = F.softmax(x_att2)
        #x_att2 = F.softmax(x_inner, axis=1).reshape([b, c, h, w])
        x_fusion = 0.5*(x_att1+x_att2)
        x_rdu = self.reduce(x)   #暂时给去掉了
        out = self.gamma * x_fusion+x_rdu


        '''
        x_qk = x_p + x_c
        x_att = paddle.multiply(x_qk, x_value) 
        # 计算attention
        '''

        out = self.huanyuan(out)
        #draw_feature_map(out)
        return out
#开始定义一个poolformer
# 定义一些初始化
trunc_normal_1 = nn.initializer.TruncatedNormal(std=0.02)
zeros_1= nn.initializer.Constant(value=0.0)
ones_1 = nn.initializer.Constant(value=1.0)

# 啥都不做的 对于torch.nn.Identity
class Identity1(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

# 下面是DropPath, 一个正则化方法
def drop_path1(x, drop_prob=0.0, training=False):

    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = paddle.to_tensor(keep_prob) + paddle.rand(shape)
    random_tensor = paddle.floor(random_tensor)
    output = x.divide(keep_prob) * random_tensor
    return output

class DropPath1(nn.Layer):
    def __init__(self, drop_prob=None):
        super(DropPath1, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path1(x, self.drop_prob, self.training)
class LayerNormChannel1(nn.Layer):
    """
    LayerNorm only for Channel Dimension.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, epsilon=1e-05):
        super().__init__()
        self.weight = paddle.create_parameter(
            shape=[num_channels],
            dtype='float32',
            default_initializer=ones_1)
        self.bias = paddle.create_parameter(
            shape=[num_channels],
            dtype='float32',
            default_initializer=zeros_1)
        self.epsilon = epsilon

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / paddle.sqrt(s + self.eps)
        x = self.weight.unsqueeze(-1).unsqueeze(-1) * x \
            + self.bias.unsqueeze(-1).unsqueeze(-1)
        return x


class GroupNorm1(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)

class Mlp1(nn.Layer):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2D(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2D(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2D):
            trunc_normal_1(m.weight)
            if m.bias is not None:
                zeros_1(m.bias)

    def forward(self, x):
        x = self.fc1(x)     # (B, C, H, W) --> (B, C, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)     # (B, C, H, W) --> (B, C, H, W)
        x = self.drop(x)
        return x
class PoolFormerBlock1(nn.Layer):
    """
    Implementation of one PoolFormer block.
    --dim: embedding dim
    --pool_size: pooling size
    --mlp_ratio: mlp expansion ratio
    --act_layer: activation
    --norm_layer: normalization
    --drop: dropout rate
    --drop path: Stochastic Depth,
        refer to https://arxiv.org/abs/1603.09382
    --use_layer_scale, --layer_scale_init_value: LayerScale,
        refer to https://arxiv.org/abs/2103.17239
    """
    def __init__(self, dim,  mlp_ratio=4.,
                 act_layer=nn.GELU, norm_layer=GroupNorm1,
                 drop=0., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5):

        super().__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer = MyAttention()   # vits是msa，MLPs是mlp，这个用pool来替代
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp1(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        # The following two techniques are useful to train deep PoolFormers.
        self.drop_path = DropPath1(drop_path) if drop_path > 0. \
            else Identity1()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:

            self.layer_scale_1 = paddle.create_parameter(
                shape=[dim],
                dtype='float32',
                default_initializer=nn.initializer.Constant(value=layer_scale_init_value))

            self.layer_scale_2 = paddle.create_parameter(
                shape=[dim],
                dtype='float32',
                default_initializer=nn.initializer.Constant(value=layer_scale_init_value))

    def forward(self, x):

        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
                * self.token_mixer(self.norm1(x)))

            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
                * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
def basic_blocks1(dim,layer,
                 pool_size=3, mlp_ratio=4.,
                 act_layer=nn.GELU, norm_layer=GroupNorm1,
                 drop_rate=.0, drop_path_rate=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5):
    """
    generate PoolFormer blocks for a stage
    return: PoolFormer blocks
    """


    block_dpr = drop_path_rate
    block=PoolFormerBlock1(
            dim,  mlp_ratio=mlp_ratio,
            act_layer=act_layer, norm_layer=norm_layer,
            drop=drop_rate, drop_path=block_dpr,
            use_layer_scale=use_layer_scale,
            layer_scale_init_value=layer_scale_init_value,
            )


    return block
class PoolFormer1(nn.Layer):
    """
    PoolFormer, the main class of our model
    --layers: [x,x,x,x], number of blocks for the 4 stages
    --embed_dims, --mlp_ratios, --pool_size: the embedding dims, mlp ratios and
        pooling size for the 4 stages
    --downsamples: flags to apply downsampling or not
    --norm_layer, --act_layer: define the types of normalizaiotn and activation
    --num_classes: number of classes for the image classification
    --in_patch_size, --in_stride, --in_pad: specify the patch embedding
        for the input image
    --down_patch_size --down_stride --down_pad:
        specify the downsample (patch embed.)
    """

    def __init__(self, layer=1,
                 embed_dims=[256],
                 mlp_ratios=[4],
                 pool_size=3,
                 norm_layer=GroupNorm1, act_layer=nn.GELU,
                 num_classes=80,
                 in_patch_size=7, in_stride=4, in_pad=2,
                 down_patch_size=3, down_stride=2, down_pad=1,
                 drop_rate=0., drop_path_rate=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 **kwargs):

        super().__init__()



        # set the main block in network
        network = []

        stage = basic_blocks1(embed_dims[0], layer=1,
                             pool_size=pool_size, mlp_ratio=mlp_ratios[0],
                             act_layer=act_layer, norm_layer=norm_layer,
                            drop_rate=drop_rate,
                            drop_path_rate=drop_path_rate,
                            use_layer_scale=use_layer_scale,
                            layer_scale_init_value=layer_scale_init_value)
        network.append(stage)
        self.network = nn.LayerList(network)

    def forward_tokens(self, x: object) -> object:
        outs = []
        for idx, block in enumerate(self.network):
            x = block(x)
        return x

    def forward(self, x):
      # through backboneF
        x = self.forward_tokens(x)
        return x

class CSPRepLayer(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=False,
                 act="silu"):
        super(CSPRepLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = BaseConv(
            in_channels, hidden_channels, ksize=1, stride=1, bias=bias, act=act)
        self.conv2 = BaseConv(
            in_channels, hidden_channels, ksize=1, stride=1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(* [
            RepVggBlock(
                hidden_channels, hidden_channels, act=act)
            for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = BaseConv(
                hidden_channels,
                out_channels,
                ksize=1,
                stride=1,
                bias=bias,
                act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)


@register
class TransformerLayer(nn.Layer):  #AIFI部分
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 attn_dropout=None,
                 act_dropout=None,
                 normalize_before=False):
        super(TransformerLayer, self).__init__()
        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout
        self.normalize_before = normalize_before

        self.self_attn = MultiHeadAttention(d_model, nhead, attn_dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(act_dropout, mode="upscale_in_train")
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout, mode="upscale_in_train")
        self.dropout2 = nn.Dropout(dropout, mode="upscale_in_train")
        self.activation = getattr(F, activation)
        self._reset_parameters()

    def _reset_parameters(self): #重新初始化可变参数
        linear_init_(self.linear1)
        linear_init_(self.linear2)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None):
        residual = src  #scr就是传入的张量
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src = self.self_attn(q, k, value=src, attn_mask=src_mask)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


@register
@serializable
class HybridEncoder(nn.Layer):
    __shared__ = ['depth_mult', 'act', 'trt', 'eval_size']  #共享的属性
#指定需要注入的属性

    def __init__(self,
                 in_channels=[128, 320, 512],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 encoder_layer='TransformerLayer',
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0, #深度倍增因子 控制编码器层的数量
                 act='silu',
                 trt=False, #是否在 TensorRT 加速模式下运行
                 eval_size=None): #指定评估过程中的输入大小
        super(HybridEncoder, self).__init__()
        self.encoder_layer=encoder_layer
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_size = eval_size
        if encoder_layer != 'TransformerLayer':
        # channel projection
            self.input_proj = nn.LayerList()
            for in_channel in in_channels:
                self.input_proj.append(
                    nn.Sequential(
                        nn.Conv2D(
                            in_channel, hidden_dim, kernel_size=1, bias_attr=False),
                        nn.BatchNorm2D(
                            hidden_dim,             #正则化     L2Decay将参数的L2范数（平方和再开方）添加到损失函数中。这会鼓励模型的权重更加平滑，即使得一些较大的权重也会被抑制，从而实现模型简化、泛化能力的提高
                            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))))
                                    #ParamAttr 表示参数属性的类 用于定义参数的各种属性
        else:
            self.input_proj = nn.LayerList()
            for in_channel in in_channels:
                self.input_proj.append(
                    nn.Sequential(
                        nn.Conv2D(
                            in_channel, hidden_dim, kernel_size=1, bias_attr=False),
                        nn.BatchNorm2D(
                            hidden_dim,
                            # 正则化     L2Decay将参数的L2范数（平方和再开方）添加到损失函数中。这会鼓励模型的权重更加平滑，即使得一些较大的权重也会被抑制，从而实现模型简化、泛化能力的提高
                            weight_attr1=ParamAttr(regularizer=L2Decay(0.0)),
                            bias_attr1=ParamAttr(regularizer=L2Decay(0.0)))))
                # ParamAttr 表示参数属性的类 用于定义参数的各种属性
        if encoder_layer!='TransformerLayer':
            self.encoder = nn.LayerList([
                PoolFormer1()
                for _ in range(len(use_encoder_idx))
            ])
        else:
            # encoder transformer
            self.encoder = nn.LayerList([
                TransformerEncoder(encoder_layer, num_encoder_layers)
                for _ in range(len(use_encoder_idx))
            ])

        act = get_act_fn(
            act, trt=trt) if act is None or isinstance(act,
                                                       (str, dict)) else act
        # top-down fpn
        self.lateral_convs = nn.LayerList()
        self.fpn_blocks = nn.LayerList()
        for idx in range(len(in_channels) - 1, 0, -1):  #2 1->S5 F4
            self.lateral_convs.append(
                BaseConv(
                    hidden_dim, hidden_dim, 1, 1, act=act))
            self.fpn_blocks.append(
                CSPRepLayer(
                    hidden_dim * 2,
                    hidden_dim,
                    round(3 * depth_mult),
                    act=act,
                    expansion=expansion))

        # bottom-up pan
        self.downsample_convs = nn.LayerList()
        self.pan_blocks = nn.LayerList()
        for idx in range(len(in_channels) - 1):
            self.downsample_convs.append(
                BaseConv(
                    hidden_dim, hidden_dim, 3, stride=2, act=act))
            self.pan_blocks.append(
                CSPRepLayer(
                    hidden_dim * 2,
                    hidden_dim,
                    round(3 * depth_mult),
                    act=act,
                    expansion=expansion))

        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_size[1] // stride, self.eval_size[0] // stride,
                    self.hidden_dim, self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w,
                                           h,
                                           embed_dim=256,
                                           temperature=10000.):
        grid_w = paddle.arange(int(w), dtype=paddle.float32)
        grid_h = paddle.arange(int(h), dtype=paddle.float32)
        grid_w, grid_h = paddle.meshgrid(grid_w, grid_h)
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = paddle.arange(pos_dim, dtype=paddle.float32) / pos_dim
        omega = 1. / (temperature**omega)

        out_w = grid_w.flatten()[..., None] @omega[None]
        out_h = grid_h.flatten()[..., None] @omega[None]

        return paddle.concat(
            [
                paddle.sin(out_w), paddle.cos(out_w), paddle.sin(out_h),
                paddle.cos(out_h)
            ],  #将四个张量在维度1上进行拼接 即将计算结果按列进行拼接
            axis=1)[None, :, :]
                            #是否用于运动目标跟踪   is_tracher  指一个已经经过训练的模型作为参考或指导。通常，教师模型在某个任务上具有较好的性能，并且可以用来提供目标值或生成预测结果，从而辅助其他模型的训练。
    def forward(self, feats, for_mot=False, is_teacher=False):

        assert len(feats) == len(self.in_channels)
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        # encoder
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):  #获取了F5
                if self.encoder_layer != 'TransformerLayer':
                    F5=proj_feats[enc_ind] #shape=[1,256,h,w]
                    S5=self.encoder[i](F5)
                    proj_feats[enc_ind] = S5
                else:
                    h, w = proj_feats[enc_ind].shape[2:]
                    # flatten [B, C, H, W] to [B, HxW, C]
                    src_flatten = proj_feats[enc_ind].flatten(2).transpose(
                        [0, 2, 1])
                    if self.training or self.eval_size is None or is_teacher:
                        pos_embed = self.build_2d_sincos_position_embedding(
                            w, h, self.hidden_dim, self.pe_temperature)
                    else:
                        pos_embed = getattr(self, f'pos_embed{enc_ind}', None)

                    memory = self.encoder[i](src_flatten, pos_embed=pos_embed)
                    proj_feats[enc_ind] = memory.transpose([0, 2, 1]).reshape(
                        [-1, self.hidden_dim, h, w])


        # top-down fpn
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](
                feat_heigh)
            inner_outs[0] = feat_heigh
            inner_outs[0] = feat_heigh

            upsample_feat = F.interpolate(
                feat_heigh, scale_factor=2., mode="nearest")
            inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](
                paddle.concat(
                    [upsample_feat, feat_low], axis=1))
            inner_outs.insert(0, inner_out)

        # bottom-up pan
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](paddle.concat(
                [downsample_feat, feat_height], axis=1))
            outs.append(out)

        return outs

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            'in_channels': [i.channels for i in input_shape],
            'feat_strides': [i.stride for i in input_shape]
        }

    @property
    def out_shape(self):
        return [
            ShapeSpec(
                channels=self.hidden_dim, stride=self.feat_strides[idx])
            for idx in range(len(self.in_channels))
        ]

def featuremap_2heatmap(feature_map):
    assert isinstance(feature_map, paddle.Tensor)
    feature_map = feature_map.detach()
    heatmap = paddle.zeros_like(feature_map[:, 0, :, :])
    for c in range(feature_map.shape[1]):
        heatmap += feature_map[:, c, :, :]
    heatmap = paddle.mean(heatmap, axis=0)
    heatmap = paddle.maximum(heatmap, paddle.zeros_like(heatmap))
    heatmap /= paddle.max(heatmap)

    return heatmap.numpy()

def draw_feature_map(features, save_dir='', name=None):
    if isinstance(features, paddle.Tensor):
        heatmaps = featuremap_2heatmap(features)
        heatmaps = cv2.resize(heatmaps, (640, 640))
        heatmap = (heatmaps * 255).astype('uint8')
       # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_RAINBOW)
        alpha = 0.5
        superimposed_img = heatmap
        plt.imshow(superimposed_img, cmap='jet',)
        plt.axis('off')  # 关闭坐标轴


        #用于获取热力图与原始图片进行结合的代码
        plt.savefig('heatmap.jpg', bbox_inches='tight', pad_inches=0)

        heatmap = Image.open('heatmap.jpg')
        original_image_path = 'D:\py12\\aim\PaddleDetection-develop\\123\photos\\2.jpg'
        original_image = Image.open(original_image_path)
        heatmap = heatmap.resize(original_image.size)
        result_image = Image.blend(original_image, heatmap, alpha)
        result_image.show()

        #plt.show()
    else:
        for featuremap in features:
            heatmaps = featuremap_2heatmap(featuremap)
            for heatmap in heatmaps:
                heatmap = (heatmap.numpy() * 255).astype('uint8')
                superimposed_img = heatmap
                plt.imshow(superimposed_img, cmap='gray')
                plt.show()
                cv2.imshow('1', superimposed_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                if save_dir and name:
                    cv2.imwrite(os.path.join(save_dir, name + '.png'), superimposed_img)