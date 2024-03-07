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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import abc
try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None



from ppdet.modeling.transformers import bbox_cxcywh_to_xyxy,bbox_xyxy_to_cxcywh
from ppdet.modeling.assigners import util_mixins
from ppdet.modeling.losses.focal_loss import FocalLoss
import paddle

from ppdet.core.workspace import register

__all__ = ['PoseHungarianAssigner', 'PseudoSampler']



class BBoxL1Cost:
    """BBoxL1Cost.

     Args:
         weight (int | float, optional): loss_weight
         box_format (str, optional): 'xyxy' for DETR, 'xywh' for Sparse_RCNN

     Examples:

         tensor([[1.6172, 1.6422]])
    """

    def __init__(self, weight=1., box_format='xyxy'):
        self.weight = weight
        assert box_format in ['xyxy', 'xywh']
        self.box_format = box_format


    def __call__(self, bbox_pred, gt_bboxes,box_format='xywh'):
        """
        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                (num_query, 4).
            gt_bboxes (Tensor): Ground truth boxes with normalized
                coordinates (x1, y1, x2, y2). Shape (num_gt, 4).

        Returns:
            torch.Tensor: bbox_cost value with weight
        """
        if box_format == 'xywh':
                gt_bboxes = self.convert_bbox(gt_bboxes, 'xywh')
        elif box_format == 'xyxy':
                bbox_pred =self.convert_bbox(bbox_pred, 'xyxy')
        bbox_cost = paddle.cdist(bbox_pred, gt_bboxes, p=1)
        return bbox_cost * 5
    def convert_bbox(self,bbox, box_format):
        if box_format == 'xywh':
            return bbox_xyxy_to_cxcywh(bbox)
        elif box_format == 'xyxy':
            return bbox_cxcywh_to_xyxy(bbox)
        else:
            raise ValueError("Invalid box format.")



def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6):
    """Calculate overlap between two set of bboxes.

    FP16 Contributed by https://github.com/open-mmlab/mmdetection/pull/4889
    Note:
        Assume bboxes1 is M x 4, bboxes2 is N x 4, when mode is 'iou',
        there are some new generated variable when calculating IOU
        using bbox_overlaps function:

        1) is_aligned is False
            area1: M x 1
            area2: N x 1
            lt: M x N x 2
            rb: M x N x 2
            wh: M x N x 2
            overlap: M x N x 1
            union: M x N x 1
            ious: M x N x 1

            Total memory:
                S = (9 x N x M + N + M) * 4 Byte,

            When using FP16, we can reduce:
                R = (9 x N x M + N + M) * 4 / 2 Byte
                R large than (N + M) * 4 * 2 is always true when N and M >= 1.
                Obviously, N + M <= N * M < 3 * N * M, when N >=2 and M >=2,
                           N + 1 < 3 * N, when N or M is 1.

            Given M = 40 (ground truth), N = 400000 (three anchor boxes
            in per grid, FPN, R-CNNs),
                R = 275 MB (one times)

            A special case (dense detection), M = 512 (ground truth),
                R = 3516 MB = 3.43 GB

            When the batch size is B, reduce:
                B x R

            Therefore, CUDA memory runs out frequently.

            Experiments on GeForce RTX 2080Ti (11019 MiB):

            |   dtype   |   M   |   N   |   Use    |   Real   |   Ideal   |
            |:----:|:----:|:----:|:----:|:----:|:----:|
            |   FP32   |   512 | 400000 | 8020 MiB |   --   |   --   |
            |   FP16   |   512 | 400000 |   4504 MiB | 3516 MiB | 3516 MiB |
            |   FP32   |   40 | 400000 |   1540 MiB |   --   |   --   |
            |   FP16   |   40 | 400000 |   1264 MiB |   276MiB   | 275 MiB |

        2) is_aligned is True
            area1: N x 1
            area2: N x 1
            lt: N x 2
            rb: N x 2
            wh: N x 2
            overlap: N x 1
            union: N x 1
            ious: N x 1

            Total memory:
                S = 11 x N * 4 Byte

            When using FP16, we can reduce:
                R = 11 x N * 4 / 2 Byte

        So do the 'giou' (large than 'iou').

        Time-wise, FP16 is generally faster than FP32.

        When gpu_assign_thr is not -1, it takes more time on cpu
        but not reduce memory.
        There, we can reduce half the memory and keep the speed.

    If ``is_aligned`` is ``False``, then calculate the overlaps between each
    bbox of bboxes1 and bboxes2, otherwise the overlaps between each aligned
    pair of bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (B, m, 4) in <x1, y1, x2, y2> format or empty.
        bboxes2 (Tensor): shape (B, n, 4) in <x1, y1, x2, y2> format or empty.
            B indicates the batch dim, in shape (B1, B2, ..., Bn).
            If ``is_aligned`` is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union), "iof" (intersection over
            foreground) or "giou" (generalized intersection over union).
            Default "iou".
        is_aligned (bool, optional): If True, then m and n must be equal.
            Default False.
        eps (float, optional): A value added to the denominator for numerical
            stability. Default 1e-6.

    Returns:
        Tensor: shape (m, n) if ``is_aligned`` is False else shape (m,)

    Example:
        >>> bboxes1 = torch.FloatTensor([
        >>>     [0, 0, 10, 10],
        >>>     [10, 10, 20, 20],
        >>>     [32, 32, 38, 42],
        >>> ])
        >>> bboxes2 = torch.FloatTensor([
        >>>     [0, 0, 10, 20],
        >>>     [0, 10, 10, 19],
        >>>     [10, 10, 20, 20],
        >>> ])
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2)
        >>> assert overlaps.shape == (3, 3)
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2, is_aligned=True)
        >>> assert overlaps.shape == (3, )

    Example:
        >>> empty = torch.empty(0, 4)
        >>> nonempty = torch.FloatTensor([[0, 0, 10, 9]])
        >>> assert tuple(bbox_overlaps(empty, nonempty).shape) == (0, 1)
        >>> assert tuple(bbox_overlaps(nonempty, empty).shape) == (1, 0)
        >>> assert tuple(bbox_overlaps(empty, empty).shape) == (0, 0)
    """

    assert mode in ['iou', 'iof', 'giou'], f'Unsupported mode {mode}'
    # Either the boxes are empty or the length of boxes' last dimension is 4
    assert (bboxes1.shape[-1] == 4 or bboxes1.shape[0] == 0)
    assert (bboxes2.shape[-1] == 4 or bboxes2.shape[0] == 0)

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.shape[-2]
    cols = bboxes2.shape[-2]
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new(batch_shape + (rows, ))
        else:
            return bboxes1.new(batch_shape + (rows, cols))

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (
        bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (
        bboxes2[..., 3] - bboxes2[..., 1])

    if is_aligned:
        lt = paddle.max(bboxes1[..., :2], bboxes2[..., :2])  # [B, rows, 2]
        rb = paddle.min(bboxes1[..., 2:], bboxes2[..., 2:])  # [B, rows, 2]

        wh = paddle.clip(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == 'giou':
            enclosed_lt = paddle.minimum(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = paddle.maximum(bboxes1[..., 2:], bboxes2[..., 2:])

    else:
        lt = paddle.maximum(bboxes1[..., :, None, :2],
                            bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
        rb = paddle.minimum(bboxes1[..., :, None, 2:],
                            bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]

        wh = paddle.clip(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1[..., None] + area2[..., None, :] - overlap
        else:
            union = area1[..., None]
        if mode == 'giou':
            enclosed_lt = paddle.minimum(bboxes1[..., :, None, :2],
                                         bboxes2[..., None, :, :2])
            enclosed_rb = paddle.maximum(bboxes1[..., :, None, 2:],
                                         bboxes2[..., None, :, 2:])

    eps = paddle.to_tensor([eps], dtype=union.dtype, place=union.place)
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


class DistillCrossEntropyLossCost:
    """CrossEntropyLossCost.

    Args:
        weight (int | float, optional): loss weight. Defaults to 1.
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
                of softmax. Defaults to True.
    Examples:

         >>> import torch
         >>> bce = CrossEntropyLossCost(use_sigmoid=True)
         >>> cls_pred = torch.tensor([[7.6, 1.2], [-1.3, 10]])
         >>> gt_labels = torch.tensor([[1, 1], [1, 0]])
         >>> print(bce(cls_pred, gt_labels))
    """

    def __init__(self, weight=1., use_sigmoid=True):
        assert use_sigmoid, 'use_sigmoid = False is not supported yet.'
        self.weight = weight
        self.use_sigmoid = use_sigmoid

    def _binary_cross_entropy(self, cls_pred, gt_labels):
        """
        Args:
            cls_pred (Tensor): The prediction with shape (num_query, 1, *) or
                (num_query, *).
            gt_labels (Tensor): The learning label of prediction with
                shape (num_gt, *).

        Returns:
            Tensor: Cross entropy cost matrix in shape (num_query, num_gt).
        """

        cls_pred = cls_pred.flatten(1).astype('float32')
        gt_labels = gt_labels.flatten(1).astype('float32')
        n = cls_pred.shape[1]

        pos = paddle.nn.functional.binary_cross_entropy_with_logits(
            cls_pred, paddle.ones_like(cls_pred),reduction='none')
        neg = paddle.nn.functional.binary_cross_entropy_with_logits(
            cls_pred, paddle.zeros_like(cls_pred), reduction='none')
        cls_cost = paddle.einsum('nc,mc->nm', pos, gt_labels) + \
                   paddle.einsum('nc,mc->nm', neg, 1 - gt_labels)
        cls_cost = cls_cost

        return cls_cost

    def __call__(self, cls_pred, gt_labels,use_sigmoid=True):
        """
        Args:
            cls_pred (Tensor): Predicted classification logits.
            gt_labels (Tensor): Labels.

        Returns:
            Tensor: Cross entropy cost matrix with weight in
                shape (num_query, num_gt).
        """
        if self.use_sigmoid:
            cls_cost = self._binary_cross_entropy(cls_pred, gt_labels)
        else:
            raise NotImplementedError

        return cls_cost * self.weight


class IoUCost:
    """IoUCost.

     Args:
         iou_mode (str, optional): iou mode such as 'iou' | 'giou'
         weight (int | float, optional): loss weight

     Examples:

         >>> self = IoUCost()
         >>> bboxes = torch.FloatTensor([[1,1, 2, 2], [2, 2, 3, 4]])
         >>> gt_bboxes = torch.FloatTensor([[0, 0, 2, 4], [1, 2, 3, 4]])
         >>> self(bboxes, gt_bboxes)
         tensor([[-0.1250,  0.1667],
                [ 0.1667, -0.5000]])
    """

    def __init__(self, iou_mode='giou', weight=1.):
        self.weight = weight
        self.iou_mode = iou_mode

    def __call__(self, bboxes, gt_bboxes,iou_mode='giou'):
        """
        Args:
            bboxes (Tensor): Predicted boxes with unnormalized coordinates
                (x1, y1, x2, y2). Shape (num_query, 4).
            gt_bboxes (Tensor): Ground truth boxes with unnormalized
                coordinates (x1, y1, x2, y2). Shape (num_gt, 4).

        Returns:
            torch.Tensor: iou_cost value with weight
        """
        # overlaps: [num_bboxes, num_gt]
        overlaps = bbox_overlaps(
            bboxes, gt_bboxes, mode=self.iou_mode,)
        # The 1 is a constant that doesn't change the matching, so omitted.
        iou_cost = -overlaps
        return iou_cost * 2
class AssignResult:
    """Stores assignments between predicted and truth boxes.

    Attributes:
        num_gts (int): the number of truth boxes considered when computing this
            assignment

        gt_inds (LongTensor): for each predicted box indicates the 1-based
            index of the assigned truth box. 0 means unassigned and -1 means
            ignore.

        max_overlaps (FloatTensor): the iou between the predicted box and its
            assigned truth box.

        labels (None | LongTensor): If specified, for each predicted box
            indicates the category label of the assigned truth box.
    """

    def __init__(self, num_gts, gt_inds, max_overlaps, labels=None):
        self.num_gts = num_gts
        self.gt_inds = gt_inds
        self.max_overlaps = max_overlaps
        self.labels = labels
        # Interface for possible user-defined properties
        self._extra_properties = {}

    @property
    def num_preds(self):
        """int: the number of predictions in this assignment"""
        return len(self.gt_inds)

    def set_extra_property(self, key, value):
        """Set user-defined new property."""
        assert key not in self.info
        self._extra_properties[key] = value

    def get_extra_property(self, key):
        """Get user-defined property."""
        return self._extra_properties.get(key, None)

    @property
    def info(self):
        """dict: a dictionary of info about the object"""
        basic_info = {
            'num_gts': self.num_gts,
            'num_preds': self.num_preds,
            'gt_inds': self.gt_inds,
            'max_overlaps': self.max_overlaps,
            'labels': self.labels,
        }
        basic_info.update(self._extra_properties)
        return basic_info

class AssignResult1(util_mixins.NiceRepr):
    """Stores assignments between predicted and truth boxes.

    Attributes:
        num_gts (int): the number of truth boxes considered when computing this
            assignment

        gt_inds (LongTensor): for each predicted box indicates the 1-based
            index of the assigned truth box. 0 means unassigned and -1 means
            ignore.

        max_overlaps (FloatTensor): the iou between the predicted box and its
            assigned truth box.

        labels (None | LongTensor): If specified, for each predicted box
            indicates the category label of the assigned truth box.

    Example:
        >>> # An assign result between 4 predicted boxes and 9 true boxes
        >>> # where only two boxes were assigned.
        >>> num_gts = 9
        >>> max_overlaps = torch.LongTensor([0, .5, .9, 0])
        >>> gt_inds = torch.LongTensor([-1, 1, 2, 0])
        >>> labels = torch.LongTensor([0, 3, 4, 0])
        >>> self = AssignResult(num_gts, gt_inds, max_overlaps, labels)
        >>> print(str(self))  # xdoctest: +IGNORE_WANT
        <AssignResult(num_gts=9, gt_inds.shape=(4,), max_overlaps.shape=(4,),
                      labels.shape=(4,))>
        >>> # Force addition of gt labels (when adding gt as proposals)
        >>> new_labels = torch.LongTensor([3, 4, 5])
        >>> self.add_gt_(new_labels)
        >>> print(str(self))  # xdoctest: +IGNORE_WANT
        <AssignResult(num_gts=9, gt_inds.shape=(7,), max_overlaps.shape=(7,),
                      labels.shape=(7,))>
    """

    def __init__(self, num_gts, gt_inds, max_overlaps, labels=None):
        self.num_gts = num_gts
        self.gt_inds = gt_inds
        self.max_overlaps = max_overlaps
        self.labels = labels
        # Interface for possible user-defined properties
        self._extra_properties = {}

    @property
    def num_preds(self):
        """int: the number of predictions in this assignment"""
        return len(self.gt_inds)

    def set_extra_property(self, key, value):
        """Set user-defined new property."""
        assert key not in self.info
        self._extra_properties[key] = value

    def get_extra_property(self, key):
        """Get user-defined property."""
        return self._extra_properties.get(key, None)

    @property
    def info(self):
        """dict: a dictionary of info about the object"""
        basic_info = {
            'num_gts': self.num_gts,
            'num_preds': self.num_preds,
            'gt_inds': self.gt_inds,
            'max_overlaps': self.max_overlaps,
            'labels': self.labels,
        }
        basic_info.update(self._extra_properties)
        return basic_info

    def __nice__(self):
        """str: a "nice" summary string describing this assign result"""
        parts = []
        parts.append(f'num_gts={self.num_gts!r}')
        if self.gt_inds is None:
            parts.append(f'gt_inds={self.gt_inds!r}')
        else:
            parts.append(f'gt_inds.shape={tuple(self.gt_inds.shape)!r}')
        if self.max_overlaps is None:
            parts.append(f'max_overlaps={self.max_overlaps!r}')
        else:
            parts.append('max_overlaps.shape='
                         f'{tuple(self.max_overlaps.shape)!r}')
        if self.labels is None:
            parts.append(f'labels={self.labels!r}')
        else:
            parts.append(f'labels.shape={tuple(self.labels.shape)!r}')
        return ', '.join(parts)

    @classmethod
    def random(cls, **kwargs):
        """Create random AssignResult for tests or debugging.

        Args:
            num_preds: number of predicted boxes
            num_gts: number of true boxes
            p_ignore (float): probability of a predicted box assigned to an
                ignored truth
            p_assigned (float): probability of a predicted box not being
                assigned
            p_use_label (float | bool): with labels or not
            rng (None | int | numpy.random.RandomState): seed or state

        Returns:
            :obj:`AssignResult`: Randomly generated assign results.

        Example:
            >>> from mmdet.core.bbox.assigners.assign_result import *  # NOQA
            >>> self = AssignResult.random()
            >>> print(self.info)
        """
        from ppdet.modeling.assigners import demodata
        rng = demodata.ensure_rng(kwargs.get('rng', None))

        num_gts = kwargs.get('num_gts', None)
        num_preds = kwargs.get('num_preds', None)
        p_ignore = kwargs.get('p_ignore', 0.3)
        p_assigned = kwargs.get('p_assigned', 0.7)
        p_use_label = kwargs.get('p_use_label', 0.5)
        num_classes = kwargs.get('p_use_label', 3)

        if num_gts is None:
            num_gts = rng.randint(0, 8)
        if num_preds is None:
            num_preds = rng.randint(0, 16)

        if num_gts == 0:
            max_overlaps = paddle.zeros(shape=[num_preds], dtype='float32')
            gt_inds = paddle.zeros(shape=[num_preds], dtype='int64')
            if p_use_label is True or p_use_label < paddle.uniform(shape=[1]):
                labels = paddle.zeros(shape=[num_preds], dtype='int64')
            else:
                labels = None
        else:
            import numpy as np

            # Create an overlap for each predicted box
            max_overlaps = paddle.to_tensor(paddle.uniform(shape=[num_preds]))
            # Construct gt_inds for each predicted box
            is_assigned = paddle.to_tensor(paddle.uniform(shape=[num_preds]) < p_assigned)
            # maximum number of assignments constraints
            n_assigned = min(num_preds, min(num_gts, is_assigned.sum()))

            assigned_idxs = np.where(is_assigned)[0]
            rng.shuffle(assigned_idxs)
            assigned_idxs = assigned_idxs[0:n_assigned]
            assigned_idxs.sort()

            is_assigned[:] = 0
            is_assigned[assigned_idxs] = True

            is_ignore = paddle.to_tensor(paddle.uniform(shape=[num_preds]) < p_ignore) & is_assigned
            gt_inds = paddle.zeros(shape=[num_preds], dtype='int64')

            true_idxs = np.arange(num_gts)
            rng.shuffle(true_idxs)
            true_idxs = paddle.to_tensor(true_idxs)
            gt_inds[is_assigned] = true_idxs[:n_assigned].long()

            gt_inds = paddle.to_tensor(paddle.randint(1, num_gts + 1, shape=[num_preds]))
            gt_inds[is_ignore] = -1
            gt_inds[~is_assigned] = 0
            max_overlaps[~is_assigned] = 0

            if p_use_label is True or p_use_label < rng.rand():
                if num_classes == 0:
                    labels = paddle.zeros(shape=[num_preds], dtype='int64')
                else:
                    labels = paddle.to_tensor(paddle.randint(0, num_classes, shape=[num_preds]))
                    labels[~is_assigned] = 0
            else:
                labels = None

        self = cls(num_gts, gt_inds, max_overlaps, labels)
        return self

    def add_gt_(self, gt_labels):
        """Add ground truth as assigned results.

        Args:
            gt_labels (torch.Tensor): Labels of gt boxes
        """
        self_inds = paddle.arange(1, len(gt_labels) + 1, dtype='int64', device=gt_labels.device)
        self.gt_inds = paddle.concat([self_inds, self.gt_inds], axis=0)

        self.max_overlaps = paddle.concat(
            [paddle.ones([len(gt_labels)], dtype=self.max_overlaps.dtype), self.max_overlaps], axis=0)

        if self.labels is not None:
            self.labels = paddle.concat([gt_labels, self.labels], axis=0)

@register
class PoseHungarianAssigner:
    """Computes one-to-one matching between predictions and ground truth.

    This class computes an assignment between the targets and the predictions
    based on the costs. The costs are weighted sum of three components:
    classification cost, regression L1 cost and regression oks cost. The
    targets don't include the no_object, so generally there are more
    predictions than targets. After the one-to-one matching, the un-matched
    are treated as backgrounds. Thus each query prediction will be assigned
    with `0` or a positive integer indicating the ground truth index:

    - 0: negative sample, no assigned gt.
    - positive integer: positive sample, index (1-based) of assigned gt.

    Args:
        cls_weight (int | float, optional): The scale factor for classification
            cost. Default 1.0.
        kpt_weight (int | float, optional): The scale factor for regression
            L1 cost. Default 1.0.
        oks_weight (int | float, optional): The scale factor for regression
            oks cost. Default 1.0.
    """
    __inject__ = ['cls_cost', 'kpt_cost', 'oks_cost']

    def __init__(self,
                 cls_cost='ClassificationCost',
                 kpt_cost='KptL1Cost',
                 oks_cost='OksCost',
                 ):
        self.cls_cost = cls_cost
        self.kpt_cost = kpt_cost
        self.oks_cost = oks_cost

    def assign(self,
               cls_pred,
               kpt_pred,
               gt_labels,
               gt_keypoints,
               gt_areas,
               img_meta,
               eps=1e-7):
        """Computes one-to-one matching based on the weighted costs.

        This method assign each query prediction to a ground truth or
        background. The `assigned_gt_inds` with -1 means don't care,
        0 means negative sample, and positive number is the index (1-based)
        of assigned gt.
        The assignment is done in the following steps, the order matters.

        1. assign every prediction to -1
        2. compute the weighted costs
        3. do Hungarian matching on CPU based on the costs
        4. assign all to 0 (background) first, then for each matched pair
           between predictions and gts, treat this prediction as foreground
           and assign the corresponding gt index (plus 1) to it.

        Args:
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            kpt_pred (Tensor): Predicted keypoints with normalized coordinates
                (x_{i}, y_{i}), which are all in range [0, 1]. Shape
                [num_query, K*2].
            gt_labels (Tensor): Label of `gt_keypoints`, shape (num_gt,).
            gt_keypoints (Tensor): Ground truth keypoints with unnormalized
                coordinates [p^{1}_x, p^{1}_y, p^{1}_v, ..., \
                    p^{K}_x, p^{K}_y, p^{K}_v]. Shape [num_gt, K*3].
            gt_areas (Tensor): Ground truth mask areas, shape (num_gt,).
            img_meta (dict): Meta information for current image.
            eps (int | float, optional): A value added to the denominator for
                numerical stability. Default 1e-7.

        Returns:
            :obj:`AssignResult`: The assigned result.
        """
        num_gts, num_kpts = gt_keypoints.shape[0], kpt_pred.shape[0]
        if not gt_keypoints.astype('bool').any():
            num_gts = 0

        # 1. assign -1 by default
        assigned_gt_inds = paddle.full((num_kpts, ), -1, dtype="int64")
        assigned_labels = paddle.full((num_kpts, ), -1, dtype="int64")
        if num_gts == 0 or num_kpts == 0:
            # No ground truth or keypoints, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)
        img_h, img_w, _ = img_meta['img_shape']
        factor = paddle.to_tensor(
            [img_w, img_h, img_w, img_h], dtype=gt_keypoints.dtype).reshape(
                (1, -1))

        # 2. compute the weighted costs
        # classification cost
        cls_cost = self.cls_cost(cls_pred, gt_labels)

        # keypoint regression L1 cost
        gt_keypoints_reshape = gt_keypoints.reshape((gt_keypoints.shape[0], -1,
                                                     3))
        valid_kpt_flag = gt_keypoints_reshape[..., -1]
        kpt_pred_tmp = kpt_pred.clone().detach().reshape((kpt_pred.shape[0], -1,
                                                          2))
        normalize_gt_keypoints = gt_keypoints_reshape[
            ..., :2] / factor[:, :2].unsqueeze(0)
        kpt_cost = self.kpt_cost(kpt_pred_tmp, normalize_gt_keypoints,
                                 valid_kpt_flag)
        # keypoint OKS cost
        kpt_pred_tmp = kpt_pred.clone().detach().reshape((kpt_pred.shape[0], -1,
                                                          2))
        kpt_pred_tmp = kpt_pred_tmp * factor[:, :2].unsqueeze(0)
        oks_cost = self.oks_cost(kpt_pred_tmp, gt_keypoints_reshape[..., :2],
                                 valid_kpt_flag, gt_areas)
        # weighted sum of above three costs
        cost = cls_cost + kpt_cost + oks_cost

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = paddle.to_tensor(matched_row_inds)
        matched_col_inds = paddle.to_tensor(matched_col_inds)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds][
            ..., 0].astype("int64")
        return AssignResult(
            num_gts, assigned_gt_inds, None, labels=assigned_labels)

    # pytoch 改 飞桨
    def assign1(self,
                bbox_pred,
                cls_pred,
                gt_bboxes,
                gt_labels,
                img_meta1,
                gt_bboxes_ignore=None,
                eps=1e-7):
        """Computes one-to-one matching based on the weighted costs.

        This method assign each query prediction to a ground truth or
        background. The `assigned_gt_inds` with -1 means don't care,
        0 means negative sample, and positive number is the index (1-based)
        of assigned gt.
        The assignment is done in the following steps, the order matters.

        1. assign every prediction to -1
        2. compute the weighted costs
        3. do Hungarian matching on CPU based on the costs
        4. assign all to 0 (background) first, then for each matched pair
           between predictions and gts, treat this prediction as foreground
           and assign the corresponding gt index (plus 1) to it.

        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                [num_query, 4].
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            gt_bboxes (Tensor): Ground truth boxes with unnormalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).
            input (dict): Meta information for current image.
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`. Default None.
            eps (int | float, optional): A value added to the denominator for
                numerical stability. Default 1e-7.

        Returns:
            :obj:`AssignResult`: The assigned result.
        """
        assert gt_bboxes_ignore is None, \
            'Only case when gt_bboxes_ignore is None is supported.'
        num_gts, num_bboxes = gt_bboxes.shape[0], bbox_pred.shape[0]
        # 1. assign -1 by default
        assigned_gt_inds = paddle.full((num_bboxes,), -1, dtype='int64')
        assigned_labels = paddle.full((num_bboxes, cls_pred.shape[1]), -1, dtype='int64')
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)
        _, _, img_h, img_w = img_meta1['image'].shape

        factor = paddle.unsqueeze(paddle.to_tensor([img_w, img_h, img_w, img_h]), axis=0)

        # 2. compute the weighted costs
        # classification and bboxcost.
        cls_cost =DistillCrossEntropyLossCost()    #pytorch中使用的蒸馏的交叉熵损失函数
        cls_cost = cls_cost(cls_pred, gt_labels)
        # regression L1 cost
        normalize_gt_bboxes = gt_bboxes / factor
        reg_cost = BBoxL1Cost()
        reg_cost = reg_cost(bbox_pred, normalize_gt_bboxes)
        # regression iou cost, defaultly giou is used in official DETR.
        bboxes = bbox_cxcywh_to_xyxy(bbox_pred) * factor
        iou_cost =IoUCost()

        iou_cost = iou_cost(bboxes, gt_bboxes)
        # weighted sum of above three costs

        cost = cls_cost + reg_cost + iou_cost

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')

        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)  #匹配的行索引和列索引
        matched_row_inds = paddle.to_tensor(matched_row_inds)
        matched_col_inds = paddle.to_tensor(matched_col_inds)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds].astype("int64")


        #assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]  #通过打印发现 gt_labels 和 assigned_lables的形状不一致 前者shape[198,80],后者shpe[198]
        return AssignResult1(
            num_gts, assigned_gt_inds, None, labels=assigned_labels)

    def assign2(self,
                    bbox_pred,
                    cls_pred,
                    gt_bboxes,
                    gt_labels,
                    img_meta1,
                    gt_bboxes_ignore=None,
                    eps=1e-7):
            """Computes one-to-one matching based on the weighted costs.

            This method assign each query prediction to a ground truth or
            background. The `assigned_gt_inds` with -1 means don't care,
            0 means negative sample, and positive number is the index (1-based)
            of assigned gt.
            The assignment is done in the following steps, the order matters.

            1. assign every prediction to -1
            2. compute the weighted costs
            3. do Hungarian matching on CPU based on the costs
            4. assign all to 0 (background) first, then for each matched pair
               between predictions and gts, treat this prediction as foreground
               and assign the corresponding gt index (plus 1) to it.

            Args:
                bbox_pred (Tensor): Predicted boxes with normalized coordinates
                    (cx, cy, w, h), which are all in range [0, 1]. Shape
                    [num_query, 4].
                cls_pred (Tensor): Predicted classification logits, shape
                    [num_query, num_class].
                gt_bboxes (Tensor): Ground truth boxes with unnormalized
                    coordinates (x1, y1, x2, y2). Shape [num_gt, 4].
                gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).
                input (dict): Meta information for current image.
                gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                    labelled as `ignored`. Default None.
                eps (int | float, optional): A value added to the denominator for
                    numerical stability. Default 1e-7.

            Returns:
                :obj:`AssignResult`: The assigned result.
            """
            assert gt_bboxes_ignore is None, \
                'Only case when gt_bboxes_ignore is None is supported.'
            num_gts, num_bboxes = gt_bboxes.shape[0], bbox_pred.shape[0]
            # 1. assign -1 by default
            assigned_gt_inds = paddle.full((num_bboxes,), -1, dtype='int64')
            assigned_labels = paddle.full((num_bboxes, cls_pred.shape[1]), -1, dtype='int64')
            if num_gts == 0 or num_bboxes == 0:
                # No ground truth or boxes, return empty assignment
                if num_gts == 0:
                    # No ground truth, assign all to background
                    assigned_gt_inds[:] = 0
                return AssignResult(
                    num_gts, assigned_gt_inds, None, labels=assigned_labels)
            _, _, img_h, img_w = img_meta1['image'].shape

            factor = paddle.unsqueeze(paddle.to_tensor([img_w, img_h, img_w, img_h]), axis=0)

            # 2. compute the weighted costs
            # classification and bboxcost.
            cls_cost = FocalLoss()  # pytorch中使用的蒸馏的交叉熵损失函数
            cls_cost = cls_cost(cls_pred, gt_labels)
            # regression L1 cost
            normalize_gt_bboxes = gt_bboxes / factor
            reg_cost = BBoxL1Cost()
            reg_cost = reg_cost(bbox_pred, normalize_gt_bboxes)
            # regression iou cost, defaultly giou is used in official DETR.
            bboxes = bbox_cxcywh_to_xyxy(bbox_pred) * factor
            iou_cost = IoUCost()

            iou_cost = iou_cost(bboxes, gt_bboxes)
            # weighted sum of above three costs
            cost = cls_cost + reg_cost + iou_cost

            # 3. do Hungarian matching on CPU using linear_sum_assignment
            cost = cost.detach().cpu()
            if linear_sum_assignment is None:
                raise ImportError('Please run "pip install scipy" '
                                  'to install scipy first.')

            matched_row_inds, matched_col_inds = linear_sum_assignment(cost)  # 匹配的行索引和列索引
            matched_row_inds = paddle.to_tensor(matched_row_inds)
            matched_col_inds = paddle.to_tensor(matched_col_inds)

            # 4. assign backgrounds and foregrounds
            # assign all indices to backgrounds first
            assigned_gt_inds[:] = 0
            # assign foregrounds based on matching results
            assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
            assigned_labels[matched_row_inds] = gt_labels[matched_col_inds].astype("int64")

            # assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]  #通过打印发现 gt_labels 和 assigned_lables的形状不一致 前者shape[198,80],后者shpe[198]
            return AssignResult1(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)
class SamplingResult:
    """Bbox sampling result.
    """

    def __init__(self, pos_inds, neg_inds, bboxes, gt_bboxes, assign_result,
                 gt_flags):

        self.pos_inds = pos_inds
        self.neg_inds = neg_inds
        if pos_inds.size > 0:
            self.pos_bboxes = bboxes[pos_inds]
            self.neg_bboxes = bboxes[neg_inds]
            self.pos_is_gt = gt_flags[pos_inds]

            self.num_gts = gt_bboxes.shape[0]
            self.pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1

            if gt_bboxes.numel() == 0:
                # hack for index error case
                assert self.pos_assigned_gt_inds.numel() == 0
                self.pos_gt_bboxes = paddle.zeros(
                    gt_bboxes.shape, dtype=gt_bboxes.dtype).reshape((-1, 4))
            else:
                if len(gt_bboxes.shape) < 2:
                    gt_bboxes = gt_bboxes.reshape((-1, 4))

                self.pos_gt_bboxes = paddle.index_select(
                    gt_bboxes,
                    self.pos_assigned_gt_inds.astype('int64'),
                    axis=0)

            if assign_result.labels is not None:
                self.pos_gt_labels = assign_result.labels[pos_inds]
            else:
                self.pos_gt_labels = None

    @property
    def bboxes(self):
        """paddle.Tensor: concatenated positive and negative boxes"""
        return paddle.concat([self.pos_bboxes, self.neg_bboxes])

    def __nice__(self):
        data = self.info.copy()
        data['pos_bboxes'] = data.pop('pos_bboxes').shape
        data['neg_bboxes'] = data.pop('neg_bboxes').shape
        parts = [f"'{k}': {v!r}" for k, v in sorted(data.items())]
        body = '    ' + ',\n    '.join(parts)
        return '{\n' + body + '\n}'

    @property
    def info(self):
        """Returns a dictionary of info about the object."""
        return {
            'pos_inds': self.pos_inds,
            'neg_inds': self.neg_inds,
            'pos_bboxes': self.pos_bboxes,
            'neg_bboxes': self.neg_bboxes,
            'pos_is_gt': self.pos_is_gt,
            'num_gts': self.num_gts,
            'pos_assigned_gt_inds': self.pos_assigned_gt_inds,
        }

class SamplingResult1(util_mixins.NiceRepr):
    """Bbox sampling result.

    Example:
        >>> # xdoctest: +IGNORE_WANT
        >>> from mmdet.core.bbox.samplers.sampling_result import *  # NOQA
        >>> self = SamplingResult.random(rng=10)
        >>> print(f'self = {self}')
        self = <SamplingResult({
            'neg_bboxes': torch.Size([12, 4]),
            'neg_inds': tensor([ 0,  1,  2,  4,  5,  6,  7,  8,  9, 10, 11, 12]),
            'num_gts': 4,
            'pos_assigned_gt_inds': tensor([], dtype=torch.int64),
            'pos_bboxes': torch.Size([0, 4]),
            'pos_inds': tensor([], dtype=torch.int64),
            'pos_is_gt': tensor([], dtype=torch.uint8)
        })>
    """

    def __init__(self, pos_inds, neg_inds, bboxes, gt_bboxes, assign_result,
                 gt_flags):
        self.pos_inds = pos_inds
        self.neg_inds = neg_inds
        self.pos_bboxes = bboxes[pos_inds]
        self.neg_bboxes = bboxes[neg_inds]

        self.pos_is_gt = gt_flags[pos_inds]  #pos_inds 是int64

        self.num_gts = gt_bboxes.shape[0]
        self.pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1

        if gt_bboxes.numel() == 0:
            # hack for index error case
            assert self.pos_assigned_gt_inds.numel() == 0
            self.pos_gt_bboxes = paddle.empty_like(gt_bboxes).reshape([-1, 4])
        else:
            if len(gt_bboxes.shape) < 2:
                gt_bboxes = gt_bboxes.view(-1, 4)

            self.pos_gt_bboxes = gt_bboxes[self.pos_assigned_gt_inds.long(), :]

        if assign_result.labels is not None:
            self.pos_gt_labels = assign_result.labels[pos_inds]
        else:
            self.pos_gt_labels = None

    @property
    def bboxes(self):
        """torch.Tensor: concatenated positive and negative boxes"""
        return paddle.concat([self.pos_bboxes, self.neg_bboxes])

    def to(self, device):
        """Change the device of the data inplace.

        Example:
            >>> self = SamplingResult.random()
            >>> print(f'self = {self.to(None)}')
            >>> # xdoctest: +REQUIRES(--gpu)
            >>> print(f'self = {self.to(0)}')
        """
        _dict = self.__dict__
        for key, value in _dict.items():
            if isinstance(value, paddle.Tensor):
                _dict[key] = value.to(device)
        return self

    def __nice__(self):
        data = self.info.copy()
        data['pos_bboxes'] = data.pop('pos_bboxes').shape
        data['neg_bboxes'] = data.pop('neg_bboxes').shape
        parts = [f"'{k}': {v!r}" for k, v in sorted(data.items())]
        body = '    ' + ',\n    '.join(parts)
        return '{\n' + body + '\n}'

    @property
    def info(self):
        """Returns a dictionary of info about the object."""
        return {
            'pos_inds': self.pos_inds,
            'neg_inds': self.neg_inds,
            'pos_bboxes': self.pos_bboxes,
            'neg_bboxes': self.neg_bboxes,
            'pos_is_gt': self.pos_is_gt,
            'num_gts': self.num_gts,
            'pos_assigned_gt_inds': self.pos_assigned_gt_inds,
        }

    @classmethod
    def random(cls, rng=None, **kwargs):
        """
        Args:
            rng (None | int | numpy.random.RandomState): seed or state.
            kwargs (keyword arguments):
                - num_preds: number of predicted boxes
                - num_gts: number of true boxes
                - p_ignore (float): probability of a predicted box assigned to \
                    an ignored truth.
                - p_assigned (float): probability of a predicted box not being \
                    assigned.
                - p_use_label (float | bool): with labels or not.

        Returns:
            :obj:`SamplingResult`: Randomly generated sampling result.

        Example:
            >>> from mmdet.core.bbox.samplers.sampling_result import *  # NOQA
            >>> self = SamplingResult.random()
            >>> print(self.__dict__)
        """
        from ppdet.modeling.assigners import demodata


        rng = demodata.ensure_rng(rng)

        # make probabalistic?
        num = 32
        pos_fraction = 0.5
        neg_pos_ub = -1

        assign_result = AssignResult1.random(rng=rng, **kwargs)

        # Note we could just compute an assignment
        bboxes = demodata.random_boxes(assign_result.num_preds, rng=rng)
        gt_bboxes = demodata.random_boxes(assign_result.num_gts, rng=rng)

        if rng.rand() > 0.2:
            # sometimes algorithms squeeze their data, be robust to that
            gt_bboxes = gt_bboxes.squeeze()
            bboxes = bboxes.squeeze()

        if assign_result.labels is None:
            gt_labels = None
        else:
            gt_labels = None  # todo

        if gt_labels is None:
            add_gt_as_proposals = False
        else:
            add_gt_as_proposals = True  # make probabalistic?

        sampler = RandomSampler(
            num,
            pos_fraction,
            neg_pos_ub=neg_pos_ub,
            add_gt_as_proposals=add_gt_as_proposals,
            rng=rng)
        self = sampler.sample(assign_result, bboxes, gt_bboxes, gt_labels)
        return self
class BaseSampler(metaclass=abc.ABCMeta):
    """Base class of samplers."""

    def __init__(self,
                 num,
                 pos_fraction,
                 neg_pos_ub=-1,
                 add_gt_as_proposals=True,
                 **kwargs):
        self.num = num
        self.pos_fraction = pos_fraction
        self.neg_pos_ub = neg_pos_ub
        self.add_gt_as_proposals = add_gt_as_proposals
        self.pos_sampler = self
        self.neg_sampler = self

    @abc.abstractmethod
    def _sample_pos(self, assign_result, num_expected, **kwargs):
        """Sample positive samples."""
        pass

    @abc.abstractmethod
    def _sample_neg(self, assign_result, num_expected, **kwargs):
        """Sample negative samples."""

    def sample(self,
               assign_result,
               bboxes,
               gt_bboxes,
               gt_labels=None,
               **kwargs):
        """Sample positive and negative bboxes.

        This is a simple implementation of bbox sampling given candidates,
        assigning results and ground truth bboxes.

        Args:
            assign_result (:obj:`AssignResult`): Bbox assigning results.
            bboxes (Tensor): Boxes to be sampled from.
            gt_bboxes (Tensor): Ground truth bboxes.
            gt_labels (Tensor, optional): Class labels of ground truth bboxes.

        Returns:
            :obj:`SamplingResult`: Sampling result.

        Example:

            >>> from ppdet.modeling.assigners.demodata import ensure_rng, random_boxes
            >>> rng = ensure_rng(None)
            >>> assign_result = AssignResult.random(rng=rng)
            >>> bboxes = random_boxes(assign_result.num_preds, rng=rng)
            >>> gt_bboxes = random_boxes(assign_result.num_gts, rng=rng)
            >>> gt_labels = None
            >>> self = RandomSampler(num=32, pos_fraction=0.5, neg_pos_ub=-1,
            >>>                      add_gt_as_proposals=False)
            >>> self = self.sample(assign_result, bboxes, gt_bboxes, gt_labels)
        """
        if len(bboxes.shape) < 2:
            bboxes = bboxes[None, :]

        bboxes = bboxes[:, :4]

        gt_flags = paddle.zeros((bboxes.shape[0],), dtype=paddle.uint8)
        if self.add_gt_as_proposals and len(gt_bboxes) > 0:
            if gt_labels is None:
                raise ValueError(
                    'gt_labels must be given when add_gt_as_proposals is True')
            bboxes = paddle.concat([gt_bboxes, bboxes], axis=0)
            assign_result.add_(gt_labels)
            gt_ones = paddle.ones(gt_bboxes.shape[0], dtype=paddle.uint8)
            gt_flags = paddle.concat([gt_ones, gt_flags], axis=0)

        num_expected_pos = int(self.num * self.pos_fraction)
        pos_inds = self.pos_sampler._sample_pos(
            assign_result, num_expected_pos, bboxes=bboxes, **kwargs)
        # We found that sampled indices have duplicated items occasionally.
        # (may be a bug of PyTorch)
        pos_inds = pos_inds.unique()
        num_sampled_pos = pos_inds.numel()
        num_expected_neg = self.num - num_sampled_pos
        if self.neg_pos_ub >= 0:
            _pos = max(1, num_sampled_pos)
            neg_upper_bound = int(self.neg_pos_ub * _pos)
            if num_expected_neg > neg_upper_bound:
                num_expected_neg = neg_upper_bound
        neg_inds = self.neg_sampler._sample_neg(
            assign_result, num_expected_neg, bboxes=bboxes, **kwargs)
        neg_inds = neg_inds.unique()

        sampling_result = SamplingResult(pos_inds, neg_inds, bboxes, gt_bboxes,
                                         assign_result, gt_flags)
        return sampling_result
class RandomSampler(BaseSampler):
    """Random sampler.

    Args:
        num (int): Number of samples
        pos_fraction (float): Fraction of positive samples
        neg_pos_up (int, optional): Upper bound number of negative and
            positive samples. Defaults to -1.
        add_gt_as_proposals (bool, optional): Whether to add ground truth
            boxes as proposals. Defaults to True.
    """

    def __init__(self,
                 num,
                 pos_fraction,
                 neg_pos_ub=-1,
                 add_gt_as_proposals=True,
                 **kwargs):
        from ppdet.modeling.assigners import demodata
        super(RandomSampler, self).__init__(num, pos_fraction, neg_pos_ub,
                                            add_gt_as_proposals)
        self.rng = demodata.ensure_rng(kwargs.get('rng', None))

    def random_choice(self, gallery, num):
        """Random select some elements from the gallery.

        If `gallery` is a Tensor, the returned indices will be a Tensor;
        If `gallery` is a ndarray or list, the returned indices will be a
        ndarray.

        Args:
            gallery (Tensor | ndarray | list): indices pool.
            num (int): expected sample num.

        Returns:
            Tensor or ndarray: sampled indices.
        """
        assert len(gallery) >= num

        is_tensor = isinstance(gallery, paddle.Tensor)
        if not is_tensor:
            if paddle.fluid.core.is_compiled_with_cuda():
                device = paddle.get_device()
            else:
                device = 'cpu'

            gallery = paddle.to_tensor(gallery, dtype='int64', place=device)
        # This is a temporary fix. We can revert the following code
        # when PyTorch fixes the abnormal return of torch.randperm.
        # See: https://github.com/open-mmlab/mmdetection/pull/5014
        perm = paddle.randperm(gallery.numel())[:num]
        rand_inds = gallery[perm]
        if not is_tensor:
            rand_inds = rand_inds.cpu().numpy()
        return rand_inds

    def _sample_pos(self, assign_result, num_expected, **kwargs):
        """Randomly sample some positive samples."""
        pos_inds = paddle.nonzero(assign_result.gt_inds > 0)
        if pos_inds.numel() != 0:
            pos_inds = pos_inds.squeeze(1)
        if pos_inds.numel() <= num_expected:
            return pos_inds
        else:
            return self.random_choice(pos_inds, num_expected)

    def _sample_neg(self, assign_result, num_expected, **kwargs):
        """Randomly sample some negative samples."""
        neg_inds = paddle.nonzero(assign_result.gt_inds == 0)
        if neg_inds.numel() != 0:
            neg_inds = neg_inds.squeeze(1)
        if len(neg_inds) <= num_expected:
            return neg_inds
        else:
            return self.random_choice(neg_inds, num_expected)
@register
class PseudoSampler:
    """A pseudo sampler that does not do sampling actually."""

    def __init__(self, **kwargs):
        pass

    def _sample_pos(self, **kwargs):
        """Sample positive samples."""
        raise NotImplementedError

    def _sample_neg(self, **kwargs):
        """Sample negative samples."""
        raise NotImplementedError

    def sample(self, assign_result, bboxes, gt_bboxes, *args, **kwargs):
        """Directly returns the positive and negative indices  of samples.

        Args:
            assign_result (:obj:`AssignResult`): Assigned results
            bboxes (paddle.Tensor): Bounding boxes
            gt_bboxes (paddle.Tensor): Ground truth boxes

        Returns:
            :obj:`SamplingResult`: sampler results
        """
        pos_inds = paddle.nonzero(
            assign_result.gt_inds > 0, as_tuple=False).squeeze(-1)

        neg_inds = paddle.nonzero(
            assign_result.gt_inds == 0, as_tuple=False).squeeze(-1)

        gt_flags = paddle.zeros([bboxes.shape[0]], dtype='uint8')
        sampling_result = SamplingResult(pos_inds, neg_inds, bboxes, gt_bboxes,
                                         assign_result, gt_flags)
        return sampling_result
    def sample1(self,assign_result, bboxes, gt_bboxes, *args, **kwargs):
        pos_inds = paddle.nonzero(
            assign_result.gt_inds > 0, as_tuple=False).squeeze(-1)
#因为蒸馏过程中 学生和老师的分类分数中不可能有负样本 因此我们选择不获取负样本
        neg_inds=None
        #neg_inds = paddle.nonzero(assign_result.gt_inds == 0, as_tuple=False).squeeze(-1)
        gt_flags = paddle.zeros((bboxes.shape[0],), dtype='int32')   #此处将uint8 改为了 int32
        sampling_result = SamplingResult(pos_inds, neg_inds, bboxes, gt_bboxes,
                                         assign_result, gt_flags)
        return sampling_result