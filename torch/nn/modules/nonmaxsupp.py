from .module import Module
from .. import functional as F

class NonMaxSuppression(Module):
    r"""Attempts to eliminate duplicate object detections by
      removing bounding boxes according to their overlap with
      higher-scoring boxes.
      
      The algorithm begins by storing the highest-scoring box and
      eliminating any box whose intersection-over-union (IoU) with
      it is too great. That procedure is repeated on the surviving
      boxes, and so on until there are no boxes left.
      The stored boxes are returned.

      Args:
          threshold (float): IoU above which to eliminate boxes

      Shape:
        - Input1 :math:`(N, n_boxes, 4)`
        - Input2 :math:`(N, n_boxes)`
        - Output1: :math:`(N, n_boxes)`
        - Output2: :math:`(N, n_boxes)`

      Examples::

      >>> nms = nn.NonMaximumSuppression(0.7)
      >>> boxes = torch.Tensor([[[10., 20., 20., 15.],
      >>>                       [24., 22., 50., 54.],
      >>>                       [10., 21., 20. 14.5]]])
      >>> scores = torch.abs(torch.randn([1, 3]))
      >>> mask, indices = nms(boxes, scores)
      >>> #indices are SORTED according to score.
      >>> surviving_boxes = indices[mask] 
      """

    def __init__(self, threshold, use_buffer=False, num_boxes=None):
        super(NonMaxSuppression, self).__init__()
        assert (threshold >= 0 and threshold <= 1), "threshold must be between 0 and 1."
        self.threshold = threshold

    def forward(self, boxes, scores):
        return F.non_max_suppression(boxes, scores, self.threshold)
