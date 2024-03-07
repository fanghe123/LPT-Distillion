import paddle
import paddle.nn.functional as F
from paddle import nn as nn

from ppdet.modeling.losses.zhengliude_loss import cross_entropy
from ppdet.core.workspace import register, create
@register
class MoCoCLHead:
    def __init__(self,
                 img_channels=256,
                 pts_channels=256,  #关键点的通道数
                 mid_channels=512,
                 img_proj_num=1,
                 pts_proj_num=1,
                 T=0.07,
                 loss_cl=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            ):
        super().__init__()
        img_projs = []; pts_projs = []
        img_input_channels = img_channels
        pts_input_channels = pts_channels
        for ii in range(img_proj_num):
            img_proj =  nn.Sequential(
                nn.Linear(img_input_channels, mid_channels),
                # nn.BatchNorm1d(mid_channels),
                # nn.ReLU(inplace=True)
            )
            img_input_channels = mid_channels
            img_projs.append(img_proj)
        for ii in range(pts_proj_num):
            pts_proj =  nn.Sequential(
                nn.Linear(pts_input_channels, mid_channels),
                # nn.BatchNorm1d(mid_channels),
                # nn.ReLU(inplace=True)
            )
            pts_input_channels = mid_channels
            pts_projs.append(pts_proj)
        self.img_projs = nn.LayerList(img_projs)
        self.pts_projs = nn.LayerList(pts_projs)
        # 2 layer mlp encoder
        self.encoder_img = nn.Sequential(
            nn.Linear(mid_channels, mid_channels),
            nn.BatchNorm1D(mid_channels),
            nn.ReLU(),
            nn.Linear(mid_channels, mid_channels)
        )
        self.encoder_pts = nn.Sequential(
            nn.Linear(mid_channels, mid_channels), nn.BatchNorm1D(mid_channels), nn.ReLU(),
            nn.Linear(mid_channels, mid_channels),
        )
        self.mid_channels = mid_channels
        self.T = T


    # @force_fp32(apply_to=('logits', 'labels'))
    # def loss(self, logits, labels):
    #     loss_cl = self.loss_cl(logits, labels)
    #     return loss_cl

    def forward(self, img_feats, pts_feats):
        for pts_proj in self.pts_projs:
            pts_feats = pts_proj(pts_feats)
        for img_proj in self.img_projs:
            img_feats = img_proj(img_feats)

        pts_feats = self.encoder_pts(pts_feats)
        pts_feats = paddle.nn.functional.normalize(pts_feats, axis=1)

        img_feats = self.encoder_img(img_feats)
        img_feats = paddle.nn.functional.normalize(img_feats, axis=1)
        # compute logits
        # Einstein sum is more intuitive
        img_feats=paddle.transpose(img_feats, perm=[1, 0])
        logits = paddle.einsum('nc,ck->nk', pts_feats, img_feats)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = paddle.arange(logits.shape[0])
        labels = paddle.to_tensor(labels)


        loss_cl = cross_entropy(logits, labels,weight=0.2)
        return loss_cl