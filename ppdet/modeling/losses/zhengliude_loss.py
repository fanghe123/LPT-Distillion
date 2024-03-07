# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register

__all__ = ['DistillCrossEntropyLoss','GIoULoss1'
]

def cross_entropy(pred,
                  label,
                  weight=None,
                  reduction='mean',
                  avg_factor=None,
                  class_weight=None,
                  ignore_index=-100,
                  avg_non_ignore=False):
    """Calculate the CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (int | None): The label index to be ignored.
            If None, it will be set to default value. Default: -100.
        avg_non_ignore (bool): The flag decides to whether the loss is
            only averaged over non-ignored targets. Default: False.

    Returns:
        torch.Tensor: The calculated loss
    """
    # The default value of ignore_index is the same as F.cross_entropy
    ignore_index = -100 if ignore_index is None else ignore_index
    # element-wise losses
    loss = F.cross_entropy(
        pred,
        label,
        weight=class_weight,
        reduction='none',
        ignore_index=ignore_index)

    # average loss over non-ignored elements
    # pytorch's official cross_entropy average loss over non-ignored elements
    # refer to https://github.com/pytorch/pytorch/blob/56b43f4fec1f76953f15a627694d4bba34588969/torch/nn/functional.py#L2660  # noqa
    if (avg_factor is None) and avg_non_ignore and reduction == 'mean':
        avg_factor = label.numel() - (label == ignore_index).sum().item()

    # apply weights and do the reduction
    if weight is not None:
        if not isinstance(weight, float):
            weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss

def weight_reduce_loss(loss, weight, reduction='mean', avg_factor=None):

    if weight is not None:
        if reduction == 'mean':
            loss = paddle.mean(loss * weight)
        elif reduction == 'sum':
            loss = paddle.sum(loss * weight)
        else:
            raise ValueError('Unsupported reduction type: {}'.format(reduction))
    elif avg_factor is not None:
        loss = loss / avg_factor
    return loss

def _expand_onehot_labels(labels, label_weights, label_channels, ignore_index):
    """Expand onehot labels to match the size of prediction."""
    bin_labels = paddle.full(size=(labels.shape[0], label_channels), fill_value=0, dtype='float32')
    valid_mask = (labels >= 0) & (labels != ignore_index)
    valid_mask = paddle.to_tensor(valid_mask)

    valid_indices = paddle.where(valid_mask & (labels < label_channels))

    if valid_indices.numel() > 0:
        bin_labels[valid_indices, labels[valid_indices]] = 1

    valid_mask = valid_mask.reshape(-1, 1).expand(labels.shape[0], label_channels).astype('float32')

    if label_weights is None:
        bin_label_weights = valid_mask
    else:
        bin_label_weights = paddle.reshape(label_weights, (-1, 1))
        bin_label_weights = paddle.tile(bin_label_weights, [1, label_channels])

        bin_label_weights *= valid_mask

    return bin_labels, bin_label_weights, valid_mask


def binary_cross_entropy(pred,
                         label,
                         weight=None,
                         reduction='mean',
                         avg_factor=None,
                         class_weight=None,
                         ignore_index=-100,
                         avg_non_ignore=False):
    """Calculate the binary CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, 1) or (N, ).
            When the shape of pred is (N, 1), label will be expanded to
            one-hot format, and when the shape of pred is (N, ), label
            will not be expanded to one-hot format.
        label (torch.Tensor): The learning label of the prediction,
            with shape (N, ).
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (int | None): The label index to be ignored.
            If None, it will be set to default value. Default: -100.
        avg_non_ignore (bool): The flag decides to whether the loss is
            only averaged over non-ignored targets. Default: False.

    Returns:
        torch.Tensor: The calculated loss.
    """
    # The default value of ignore_index is the same as F.cross_entropy
    ignore_index = -100 if ignore_index is None else ignore_index

    """
    if pred.dim() != label.dim():
        label, weight, valid_mask = _expand_onehot_labels(
            label, weight, pred.size(-1), ignore_index)
    else:
        # should mask out the ignored elements
        valid_mask = ((label >= 0) & (label != ignore_index)).float()
        if weight is not None:
            # The inplace writing method will have a mismatched broadcast
            # shape error if the weight and valid_mask dimensions
            # are inconsistent such as (B,N,1) and (B,N,C).
            weight = weight * valid_mask
        else:
            weight = valid_mask

    # average loss over non-ignored elements
    if (avg_factor is None) and avg_non_ignore and reduction == 'mean':
        avg_factor = valid_mask.sum().item()
    """
    """
    ignore_index = -100 if ignore_index is None else ignore_index

    if pred.dim() != label.dim():
        label, weight, valid_mask = _expand_onehot_labels(
            label, weight, pred.size(-1), ignore_index)
    else:
        # should mask out the ignored elements
        valid_mask = paddle.cast((label >= 0) & (label != ignore_index), dtype='float32')
        if weight is not None:
            # The inplace writing method will have a mismatched broadcast
            # shape error if the weight and valid_mask dimensions
            # are inconsistent such as (B,N,1) and (B,N,C).
            weight = weight * valid_mask
        else:
            weight = valid_mask

    # average loss over non-ignored elements
    if (avg_factor is None) and avg_non_ignore and reduction == 'mean':
        avg_factor = valid_mask.sum().item()
"""
    # weighted element-wise losses
    weight = weight.astype('float32')
    loss = F.binary_cross_entropy_with_logits(
        pred,paddle.cast(label, 'float32'), pos_weight=class_weight, reduction='none')
    # do the reduction for the weighted loss
    loss = weight_reduce_loss(
        loss, weight, reduction=reduction, avg_factor=avg_factor)

    return loss


@register
class DistillCrossEntropyLoss(nn.Layer):

    def __init__(self,
                 use_sigmoid=True,
                 use_mask=False,
                 reduction='mean',
                 class_weight=None,
                 ignore_index=None,
                 loss_weight=1.0,
                 avg_non_ignore=False):
        """CrossEntropyLoss.

        Args:
            use_sigmoid (bool, optional): Whether the prediction uses sigmoid
                of softmax. Defaults to False.
            use_mask (bool, optional): Whether to use mask cross entropy loss.
                Defaults to False.
            reduction (str, optional): . Defaults to 'mean'.
                Options are "none", "mean" and "sum".
            class_weight (list[float], optional): Weight of each class.
                Defaults to None.
            ignore_index (int | None): The label index to be ignored.
                Defaults to None.
            loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
            avg_non_ignore (bool): The flag decides to whether the loss is
                only averaged over non-ignored targets. Default: False.
        """
        super(DistillCrossEntropyLoss, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.ignore_index = ignore_index
        self.avg_non_ignore = avg_non_ignore
        if ((ignore_index is not None) and not self.avg_non_ignore
                and self.reduction == 'mean'):
            warnings.warn(
                'Default ``avg_non_ignore`` is False, if you would like to '
                'ignore the certain label and average loss over non-ignore '
                'labels, which is the same with PyTorch official '
                'cross_entropy, set ``avg_non_ignore=True``.')

        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        else:
            self.cls_criterion = cross_entropy

    def extra_repr(self):
        """Extra repr."""
        s = f'avg_non_ignore={self.avg_non_ignore}'
        return s

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=None,
                **kwargs):
        """Forward function.

        Args:
            cls_score (torch.Tensor): The prediction.
            label (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): Sample-wise loss weight.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The method used to reduce the
                loss. Options are "none", "mean" and "sum".
            ignore_index (int | None): The label index to be ignored.
                If not None, it will override the default value. Default: None.
        Returns:
            torch.Tensor: The calculated loss.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction='mean'
        reduction = (
            reduction_override if reduction_override else reduction)
        if ignore_index is None:
            ignore_index = None

        if self.class_weight is not None:
            class_weight = paddle.to_tensor(self.class_weight, place=cls_score.place)
        else:
            class_weight = None
        loss_cls = 1.0 * binary_cross_entropy(
            cls_score,
            label,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            ignore_index=ignore_index,
            avg_non_ignore=False,
            **kwargs)
        return loss_cls

@register
class GIoULoss1(nn.Layer):

    def __init__(self, eps=1e-6, reduction='mean', loss_weight=1.0):
        super(GIoULoss1, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        reduction='mean'
        if weight is not None and not paddle.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else reduction)
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # giou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = 2.0 * giou_loss(
            pred,
            target,
            eps=1e-6,   #根据函数做出了相应的传入参数的修改
            **kwargs)
        return loss

def giou_loss(pred, target, eps=1e-7):
    r"""`Generalized Intersection over Union: A Metric and A Loss for Bounding
    Box Regression <https://arxiv.org/abs/1902.09630>`_.

    Args:
        pred (torch.Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (torch.Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).

    Return:
        Tensor: Loss tensor.
    """
    gious = bbox_overlaps(pred, target, mode='giou', is_aligned=True, eps=eps)
    loss = 1 - gious
    return loss

def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6):
    if mode not in ['iou', 'iof', 'giou']:
        raise AssertionError(f'Unsupported mode {mode}')
    # Either the boxes are empty or the length of boxes' last dimension is 4
    if not (bboxes1.shape[-1] == 4 or bboxes1.shape[0] == 0):
        raise AssertionError('Either the last dimension of bboxes1 should be 4 or the first dimension should be 0')

    if not (bboxes2.shape[-1] == 4 or bboxes2.shape[0] == 0):
        raise AssertionError('Either the last dimension of bboxes2 should be 4 or the first dimension should be 0')

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    if bboxes1.shape[:-2] != bboxes2.shape[:-2]:
        raise AssertionError('The shapes of bboxes1 and bboxes2 should match in all dimensions except the last two.')
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.shape[-2]
    cols = bboxes2.shape[-2]
    if is_aligned:
        if rows != cols:
            raise AssertionError('Rows and columns should be equal when is_aligned is True.')

    if rows * cols == 0:
        if is_aligned:
            return paddle.full(shape=batch_shape + [rows], fill_value=0, dtype=bboxes1.dtype)
        else:
            return paddle.full(shape=batch_shape + [rows, cols], fill_value=0, dtype=bboxes1.dtype)

    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])

    if is_aligned:
        lt = paddle.maximum(bboxes1[:, :2], bboxes2[:, :2])  # [B, rows, 2]
        rb = paddle.minimum(bboxes1[:, 2:], bboxes2[:, 2:])  # [B, rows, 2]

        wh = paddle.clip(rb - lt, min=0)
        overlap = wh[:, 0] * wh[:, 1]

        if mode in ['iou', 'giou']:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == 'giou':
            enclosed_lt = paddle.minimum(bboxes1[:, :2], bboxes2[:, :2])
            enclosed_rb = paddle.maximum(bboxes1[:, 2:], bboxes2[:, 2:])
    else:
            lt = paddle.maximum(bboxes1[:, :, None, :2], bboxes2[:, None, :, :2])  # [B, rows, cols, 2]
            rb = paddle.minimum(bboxes1[:, :, None, 2:], bboxes2[:, None, :, 2:])  # [B, rows, cols, 2]

            wh = paddle.clip(rb - lt, min=0)
            overlap = wh[:, :, 0] * wh[:, :, 1]

            if mode in ['iou', 'giou']:
                    union = area1[:, None] + area2[:, None, :] - overlap
            else:
                    union = area1[:, None]
            if mode == 'giou':
                    enclosed_lt = paddle.minimum(bboxes1[:, :, None, :2], bboxes2[:, None, :, :2])
                    enclosed_rb = paddle.maximum(bboxes1[:, :, None, 2:], bboxes2[:, None, :, 2:])

    eps = paddle.to_tensor([eps])
    union = paddle.maximum(union, eps)
    ious = overlap / union
    if mode in ['iou', 'iof']:
        return ious
    # calculate gious
    enclose_wh = paddle.clip(enclosed_rb - enclosed_lt, min=0)

    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    enclose_area = paddle.maximum(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious