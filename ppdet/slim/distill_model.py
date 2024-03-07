# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import paddle.nn as nn
from ppdet.modeling.heads.moco_cl_head import MoCoCLHead
from ppdet.core.workspace import register, create, load_config
from ppdet.utils.checkpoint import load_pretrain_weight
from ppdet.utils.logger import setup_logger
from ppdet.modeling.transformers.position_encoding import PositionEmbedding
logger = setup_logger(__name__)

__all__ = [
    'DETRDistillModel',
    'DistillModel',
    'FGDDistillModel',
    'CWDDistillModel',
    'LDDistillModel',
    'PPYOLOEDistillModel',
]

@register
class DETRDistillModel(nn.Layer):
    """
    Build common distill model.
    Args:
        cfg: The student config.
        slim_cfg: The teacher and distill config.
    """

    def __init__(self, cfg, slim_cfg,
                 init_student=False,
                 teacher_pretrained=None,
                 is_layer_by_layer_distill=True,

                 ):
        #使用了交叉熵损失函数):
        super(DETRDistillModel, self).__init__()
        self.arch = cfg.architecture
        assert self.arch in ['DETR'], 'Unsupported arch: {}'.format(self.arch)

        self.stu_cfg = cfg
        self.student_model = create(self.stu_cfg.architecture)
        if 'pretrain_weights' in self.stu_cfg and self.stu_cfg.pretrain_weights:
            stu_pretrain = self.stu_cfg.pretrain_weights
        else:
            stu_pretrain = None

        slim_cfg = load_config(slim_cfg)  #获取教师的配置并构建教师模型
        self.tea_cfg = slim_cfg
        self.teacher_model = create(self.tea_cfg.architecture)
        if 'pretrain_weights' in self.tea_cfg and self.tea_cfg.pretrain_weights:
            tea_pretrain = self.tea_cfg.pretrain_weights
        else:
            tea_pretrain = None

        self.distill_cfg = slim_cfg

        #加载教师和学生的预训练权重
        # load pretrain weights
        self.is_inherit = False
        if stu_pretrain:
            if self.is_inherit and tea_pretrain:
                load_pretrain_weight(self.student_model, tea_pretrain)
                logger.debug(
                    "Inheriting! loading teacher weights to student model!")
            load_pretrain_weight(self.student_model, stu_pretrain)
            logger.info("Student model has loaded pretrain weights!")
        if tea_pretrain:
            load_pretrain_weight(self.teacher_model, tea_pretrain)
            logger.info("Teacher model has loaded pretrain weights!")

        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.stop_gradient = True
            param.trainable = False

        self.is_layer_by_layer_distill = is_layer_by_layer_distill

        self.cl_head= MoCoCLHead()
        self.distill_loss = self.build_loss(self.distill_cfg)

    def build_loss(self, distill_cfg):
        if 'distill_loss' in distill_cfg and distill_cfg.distill_loss:
            return create(distill_cfg.distill_loss)
        else:
            return None

    def parameters(self):
        return self.student_model.parameters()

    def forward(self, inputs):
        if self.training:
            student_loss = self.student_model(inputs)
            with paddle.no_grad():
                teacher_loss = self.teacher_model(inputs)

            loss,distill_class_loss, distill_box_loss,distill_iou_loss,teacher_relation,student_relation = self.distill_loss(self.teacher_model, self.student_model,inputs)   #意味着我们进需要获取decoder部分层和层之间的蒸馏即可
            student_loss['distill_loss'] = loss
            student_loss['distill_class_loss']=distill_class_loss
            student_loss['distill_box_loss'] = distill_box_loss
            student_loss['distill_iou_loss'] = distill_iou_loss
            student_loss['teacher_loss'] = teacher_loss['loss']
            student_loss['loss'] += student_loss['distill_loss']
            teacher_all_stage_det_querys, all_stage_cls_iou_score, all_stage_weight_query, query_embedding,tea_feat=teacher_relation
            student_all_stage_det_querys, all_stage_pos_assigned_gt_inds,student_bboxes,student_class,student_features,gt_bbox, gt_class=student_relation

            stages = len(all_stage_cls_iou_score)  # all_stage_cls_iou_score 来自教师模型
            loss_mse = nn.MSELoss(reduction='mean')
            conv = paddle.nn.Conv2D(in_channels=384, out_channels=256, kernel_size=1)
            teacher_features1 = []
            for tea_feat1 in tea_feat:
                teacher = conv(tea_feat1)
                teacher_features1.append(teacher)
            for stage in range(stages):
                stage_cls_iou_score = all_stage_cls_iou_score[stage]
                stage_cls_iou_score = paddle.stack(stage_cls_iou_score, axis=0)
                teacher_stage_det_querys = all_stage_weight_query[stage]
                teacher_stage_det_querys = paddle.stack(teacher_stage_det_querys, axis=0)
                stage_loss = 0
                teacher_features = teacher_features1
                # c_querys = torch.nn.functional.normalize(self.domain_covn(teacher_stage_det_querys), dim=-1)
                c_querys = paddle.nn.functional.normalize(teacher_stage_det_querys, p=2, axis=-1)  # 进行L2范数归一化
                # 对教师模型和学生模型进行特征蒸馏

                for fea_tea, fea_stu in zip(teacher_features, student_features):
                    c_feats = paddle.nn.functional.normalize(fea_tea, p=2, axis=1)

                    mat = paddle.einsum('bnc,bchw->bnhw', c_querys, c_feats)

                    mask = paddle.einsum('bnhw,bn->bhw', mat, stage_cls_iou_score).clip(min=1e-2)

                    max_shu = paddle.max(mask.flatten(1,2),axis=-1)[0].unsqueeze(axis=-1).unsqueeze(axis=-1)

                    mask = (mask / max_shu).unsqueeze(axis=1)
                    stage_loss += loss_mse(mask * fea_tea, mask * fea_stu)

                    student_loss[f'stage{stage}_relation_distill_loss'] = stage_loss

            #是对query进行蒸馏
            if self.cl_head is not None:
                # TODO: if use the all 6 stages query
                stages = len(teacher_all_stage_det_querys)
                for stage in range(stages):
                    teacher_det_querys_stage = teacher_all_stage_det_querys[stage]
                    student_det_querys_distill_stage = student_all_stage_det_querys[stage]

                    pos_assigned_gt_inds_stage = all_stage_pos_assigned_gt_inds[stage]
                    pos_assigned_gt_inds_stage=[pos_assigned_gt_inds_stage[0],pos_assigned_gt_inds_stage[1]]

                    teacher_det_querys_distill_stage = [teacher_det_querys_stage[i][index] for i, index in enumerate(pos_assigned_gt_inds_stage)]

                    #student_det_querys_distill_stage = paddle.concat(student_det_querys_distill_stage, axis=1)
                    b, n, d = student_det_querys_distill_stage.shape

                    student_det_querys_distill_stage = paddle.reshape(student_det_querys_distill_stage,shape=[b * n, d])
                    teacher_det_querys_distill_stage = paddle.concat(teacher_det_querys_distill_stage)


                    relation_distill_loss_stage = self.cl_head.forward(student_det_querys_distill_stage,
                                                                       teacher_det_querys_distill_stage)
                    student_loss[f'stage{stage}_relation_distill_loss'] += relation_distill_loss_stage

            return student_loss
        else:
            return self.student_model(inputs)

@register
class DistillModel(nn.Layer):
    """
    Build common distill model.
    Args:
        cfg: The student config.
        slim_cfg: The teacher and distill config.
    """

    def __init__(self, cfg, slim_cfg):
        super(DistillModel, self).__init__()
        self.arch = cfg.architecture

        self.stu_cfg = cfg
        self.student_model = create(self.stu_cfg.architecture)
        if 'pretrain_weights' in self.stu_cfg and self.stu_cfg.pretrain_weights:
            stu_pretrain = self.stu_cfg.pretrain_weights
        else:
            stu_pretrain = None

        slim_cfg = load_config(slim_cfg)
        self.tea_cfg = slim_cfg
        self.teacher_model = create(self.tea_cfg.architecture)
        if 'pretrain_weights' in self.tea_cfg and self.tea_cfg.pretrain_weights:
            tea_pretrain = self.tea_cfg.pretrain_weights
        else:
            tea_pretrain = None
        self.distill_cfg = slim_cfg

        # load pretrain weights
        self.is_inherit = False
        if stu_pretrain:
            if self.is_inherit and tea_pretrain:
                load_pretrain_weight(self.student_model, tea_pretrain)
                logger.debug(
                    "Inheriting! loading teacher weights to student model!")
            load_pretrain_weight(self.student_model, stu_pretrain)
            logger.info("Student model has loaded pretrain weights!")
        if tea_pretrain:
            load_pretrain_weight(self.teacher_model, tea_pretrain)
            logger.info("Teacher model has loaded pretrain weights!")

        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.trainable = False

        self.distill_loss = self.build_loss(self.distill_cfg)

    def build_loss(self, distill_cfg):
        if 'distill_loss' in distill_cfg and distill_cfg.distill_loss:
            return create(distill_cfg.distill_loss)
        else:
            return None

    def parameters(self):
        return self.student_model.parameters()

    def forward(self, inputs):
        if self.training:
            student_loss = self.student_model(inputs)
            with paddle.no_grad():
                teacher_loss = self.teacher_model(inputs)

            loss = self.distill_loss(self.teacher_model, self.student_model,inputs)
            student_loss['distill_loss'] = loss
            student_loss['teacher_loss'] = teacher_loss['loss']
            student_loss['loss'] += student_loss['distill_loss']

            return student_loss
        else:
            return self.student_model(inputs)
''

@register
class FGDDistillModel(DistillModel):
    """
    Build FGD distill model.
    Args:
        cfg: The student config.
        slim_cfg: The teacher and distill config.
    """

    def __init__(self, cfg, slim_cfg):
        super(FGDDistillModel, self).__init__(cfg=cfg, slim_cfg=slim_cfg)
        assert self.arch in ['RetinaNet', 'PicoDet'
                             ], 'Unsupported arch: {}'.format(self.arch)
        self.is_inherit = True

    def build_loss(self, distill_cfg):
        assert 'distill_loss_name' in distill_cfg and distill_cfg.distill_loss_name
        assert 'distill_loss' in distill_cfg and distill_cfg.distill_loss
        loss_func = dict()
        name_list = distill_cfg.distill_loss_name
        for name in name_list:
            loss_func[name] = create(distill_cfg.distill_loss)
        return loss_func

    def forward(self, inputs):
        if self.training:
            s_body_feats = self.student_model.backbone(inputs)
            s_neck_feats = self.student_model.neck(s_body_feats)
            with paddle.no_grad():
                t_body_feats = self.teacher_model.backbone(inputs)
                t_neck_feats = self.teacher_model.neck(t_body_feats)

            loss_dict = {}
            for idx, k in enumerate(self.distill_loss):
                loss_dict[k] = self.distill_loss[k](s_neck_feats[idx],
                                                    t_neck_feats[idx], inputs)
            if self.arch == "RetinaNet":
                loss = self.student_model.head(s_neck_feats, inputs)
            elif self.arch == "PicoDet":
                head_outs = self.student_model.head(
                    s_neck_feats, self.student_model.export_post_process)
                loss_gfl = self.student_model.head.get_loss(head_outs, inputs)
                total_loss = paddle.add_n(list(loss_gfl.values()))
                loss = {}
                loss.update(loss_gfl)
                loss.update({'loss': total_loss})
            else:
                raise ValueError(f"Unsupported model {self.arch}")

            for k in loss_dict:
                loss['loss'] += loss_dict[k]
                loss[k] = loss_dict[k]
            return loss
        else:
            body_feats = self.student_model.backbone(inputs)
            neck_feats = self.student_model.neck(body_feats)
            head_outs = self.student_model.head(neck_feats)
            if self.arch == "RetinaNet":
                bbox, bbox_num = self.student_model.head.post_process(
                    head_outs, inputs['im_shape'], inputs['scale_factor'])
                return {'bbox': bbox, 'bbox_num': bbox_num}
            elif self.arch == "PicoDet":
                head_outs = self.student_model.head(
                    neck_feats, self.student_model.export_post_process)
                scale_factor = inputs['scale_factor']
                bboxes, bbox_num = self.student_model.head.post_process(
                    head_outs,
                    scale_factor,
                    export_nms=self.student_model.export_nms)
                return {'bbox': bboxes, 'bbox_num': bbox_num}
            else:
                raise ValueError(f"Unsupported model {self.arch}")


@register
class CWDDistillModel(DistillModel):
    """
    Build CWD distill model.
    Args:
        cfg: The student config.
        slim_cfg: The teacher and distill config.
    """

    def __init__(self, cfg, slim_cfg):
        super(CWDDistillModel, self).__init__(cfg=cfg, slim_cfg=slim_cfg)
        assert self.arch in ['GFL', 'RetinaNet'], 'Unsupported arch: {}'.format(
            self.arch)

    def build_loss(self, distill_cfg):
        assert 'distill_loss_name' in distill_cfg and distill_cfg.distill_loss_name
        assert 'distill_loss' in distill_cfg and distill_cfg.distill_loss
        loss_func = dict()
        name_list = distill_cfg.distill_loss_name
        for name in name_list:
            loss_func[name] = create(distill_cfg.distill_loss)
        return loss_func

    def get_loss_retinanet(self, stu_fea_list, tea_fea_list, inputs):
        loss = self.student_model.head(stu_fea_list, inputs)
        loss_dict = {}
        for idx, k in enumerate(self.distill_loss):
            loss_dict[k] = self.distill_loss[k](stu_fea_list[idx],
                                                tea_fea_list[idx])

            loss['loss'] += loss_dict[k]
            loss[k] = loss_dict[k]
        return loss

    def get_loss_gfl(self, stu_fea_list, tea_fea_list, inputs):
        loss = {}
        head_outs = self.student_model.head(stu_fea_list)
        loss_gfl = self.student_model.head.get_loss(head_outs, inputs)
        loss.update(loss_gfl)
        total_loss = paddle.add_n(list(loss.values()))
        loss.update({'loss': total_loss})

        feat_loss = {}
        loss_dict = {}
        s_cls_feat, t_cls_feat = [], []
        for s_neck_f, t_neck_f in zip(stu_fea_list, tea_fea_list):
            conv_cls_feat, _ = self.student_model.head.conv_feat(s_neck_f)
            cls_score = self.student_model.head.gfl_head_cls(conv_cls_feat)
            t_conv_cls_feat, _ = self.teacher_model.head.conv_feat(t_neck_f)
            t_cls_score = self.teacher_model.head.gfl_head_cls(t_conv_cls_feat)
            s_cls_feat.append(cls_score)
            t_cls_feat.append(t_cls_score)

        for idx, k in enumerate(self.distill_loss):
            loss_dict[k] = self.distill_loss[k](s_cls_feat[idx],
                                                t_cls_feat[idx])
            feat_loss[f"neck_f_{idx}"] = self.distill_loss[k](stu_fea_list[idx],
                                                              tea_fea_list[idx])

        for k in feat_loss:
            loss['loss'] += feat_loss[k]
            loss[k] = feat_loss[k]

        for k in loss_dict:
            loss['loss'] += loss_dict[k]
            loss[k] = loss_dict[k]
        return loss

    def forward(self, inputs):
        if self.training:
            s_body_feats = self.student_model.backbone(inputs)
            s_neck_feats = self.student_model.neck(s_body_feats)
            with paddle.no_grad():
                t_body_feats = self.teacher_model.backbone(inputs)
                t_neck_feats = self.teacher_model.neck(t_body_feats)

            if self.arch == "RetinaNet":
                loss = self.get_loss_retinanet(s_neck_feats, t_neck_feats,
                                               inputs)
            elif self.arch == "GFL":
                loss = self.get_loss_gfl(s_neck_feats, t_neck_feats, inputs)
            else:
                raise ValueError(f"unsupported arch {self.arch}")
            return loss
        else:
            body_feats = self.student_model.backbone(inputs)
            neck_feats = self.student_model.neck(body_feats)
            head_outs = self.student_model.head(neck_feats)
            if self.arch == "RetinaNet":
                bbox, bbox_num = self.student_model.head.post_process(
                    head_outs, inputs['im_shape'], inputs['scale_factor'])
                return {'bbox': bbox, 'bbox_num': bbox_num}
            elif self.arch == "GFL":
                bbox_pred, bbox_num = head_outs
                output = {'bbox': bbox_pred, 'bbox_num': bbox_num}
                return output
            else:
                raise ValueError(f"unsupported arch {self.arch}")


@register
class LDDistillModel(DistillModel):
    """
    Build LD distill model.
    Args:
        cfg: The student config.
        slim_cfg: The teacher and distill config.
    """

    def __init__(self, cfg, slim_cfg):
        super(LDDistillModel, self).__init__(cfg=cfg, slim_cfg=slim_cfg)
        assert self.arch in ['GFL'], 'Unsupported arch: {}'.format(self.arch)

    def forward(self, inputs):
        if self.training:
            s_body_feats = self.student_model.backbone(inputs)
            s_neck_feats = self.student_model.neck(s_body_feats)
            s_head_outs = self.student_model.head(s_neck_feats)
            with paddle.no_grad():
                t_body_feats = self.teacher_model.backbone(inputs)
                t_neck_feats = self.teacher_model.neck(t_body_feats)
                t_head_outs = self.teacher_model.head(t_neck_feats)

            soft_label_list = t_head_outs[0]
            soft_targets_list = t_head_outs[1]
            student_loss = self.student_model.head.get_loss(
                s_head_outs, inputs, soft_label_list, soft_targets_list)
            total_loss = paddle.add_n(list(student_loss.values()))
            student_loss['loss'] = total_loss
            return student_loss
        else:
            return self.student_model(inputs)


@register
class PPYOLOEDistillModel(DistillModel):
    """
    Build PPYOLOE distill model, only used in PPYOLOE
    Args:
        cfg: The student config.
        slim_cfg: The teacher and distill config.
    """

    def __init__(self, cfg, slim_cfg):
        super(PPYOLOEDistillModel, self).__init__(cfg=cfg, slim_cfg=slim_cfg)
        assert self.arch in ['PPYOLOE'], 'Unsupported arch: {}'.format(
            self.arch)

    def forward(self, inputs, alpha=0.125):
        if self.training:
            with paddle.no_grad():
                teacher_loss = self.teacher_model(inputs)
            if hasattr(self.teacher_model.yolo_head, "assigned_labels"):
                self.student_model.yolo_head.assigned_labels, self.student_model.yolo_head.assigned_bboxes, self.student_model.yolo_head.assigned_scores = \
                    self.teacher_model.yolo_head.assigned_labels, self.teacher_model.yolo_head.assigned_bboxes, self.teacher_model.yolo_head.assigned_scores
                delattr(self.teacher_model.yolo_head, "assigned_labels")
                delattr(self.teacher_model.yolo_head, "assigned_bboxes")
                delattr(self.teacher_model.yolo_head, "assigned_scores")
            student_loss = self.student_model(inputs)

            logits_loss, feat_loss = self.distill_loss(self.teacher_model,
                                                       self.student_model)
            det_total_loss = student_loss['loss']
            total_loss = alpha * (det_total_loss + logits_loss + feat_loss)
            student_loss['loss'] = total_loss
            student_loss['det_loss'] = det_total_loss
            student_loss['logits_loss'] = logits_loss
            student_loss['feat_loss'] = feat_loss
            return student_loss
        else:
            return self.student_model(inputs)

