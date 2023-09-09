import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
import numpy as np


class LargeMarginSoftmax(nn.CrossEntropyLoss):
    """
    This combines the Softmax Cross-Entropy Loss (nn.CrossEntropyLoss) and the large-margin inducing
    regularization proposed in
       T. Kobayashi, "Large-Margin In Softmax Cross-Entropy Loss." In BMVC2019.
    """

    def __init__(self, reg_lambda=0.3, deg_logit=None,
                 weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        super(LargeMarginSoftmax, self).__init__(weight=weight, size_average=size_average,
                                                 ignore_index=ignore_index, reduce=reduce, reduction=reduction)
        self.reg_lambda = reg_lambda
        self.deg_logit = deg_logit

    def forward(self, input, target):
        N = input.size(0)  # number of samples
        C = input.size(1)  # number of classes
        Mask = torch.zeros_like(input, requires_grad=False)
        Mask[range(N), target] = 1

        if self.deg_logit is not None:
            input = input - self.deg_logit * Mask

        loss = F.cross_entropy(input, target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)

        X = input - 1.e6 * Mask  # [N x C], excluding the target class
        reg = 0.5 * ((F.softmax(X, dim=1) - 1.0 / (C - 1)) * F.log_softmax(X, dim=1) * (1.0 - Mask)).sum(dim=1)
        if self.reduction == 'sum':
            reg = reg.sum()
        elif self.reduction == 'mean':
            reg = reg.mean()
        elif self.reduction == 'none':
            reg = reg

        return loss + self.reg_lambda * reg


class total_LargeMargin_CrossEntropy(nn.Module):
    def __init__(self):
        super(total_LargeMargin_CrossEntropy, self).__init__()
        self.loss1 = LargeMarginSoftmax()
        self.loss2 = LargeMarginSoftmax()

    def forward(self, s1: torch.Tensor, s2: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        s1_loss = self.loss1(s1, target)
        s2_loss = self.loss2(s2, target)

        total_loss = s1_loss + s2_loss

        return total_loss


class CFPC_loss(nn.Module):
    """
    This combines the CrossCLR Loss proposed in
       M. Zolfaghari et al., "CrossCLR: Cross-modal Contrastive Learning For Multi-modal Video Representations,"
       In ICCV2021.
    """

    def __init__(self, temperature=0.02, negative_weight=0.8, config=None):
        super(CFPC_loss, self).__init__()
        self.logit_scale = nn.Parameter(torch.ones([]))
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.temperature = temperature
        self.config = config
        self.negative_w = negative_weight  # Weight of negative samples logits.

    def compute_loss(self, logits, mask):
        return - torch.log((F.softmax(logits, dim=1) * mask).sum(1))

    def _get_positive_mask(self, batch_size):
        diag = np.eye(batch_size)
        mask = torch.from_numpy(diag)
        mask = (1 - mask)
        return mask.to(self.config.device)

    def forward(self, face_features, ocular_features):
        """
        Inputs shape (batch, embed_dim)
        Args:
            face_features: face embeddings (batch, embed_dim)
            ocular_features: ocular embeddings (batch, embed_dim)
        Returns:
        """
        batch_size = face_features.shape[0]

        # Normalize features
        face_features = nn.functional.normalize(face_features, dim=1)
        ocular_features = nn.functional.normalize(ocular_features, dim=1)

        # Inter-modality alignment
        logits_per_face = face_features @ ocular_features.t()
        logits_per_ocular = ocular_features @ face_features.t()

        # Intra-modality alignment
        logits_clstr_face = face_features @ face_features.t()
        logits_clstr_ocular = ocular_features @ ocular_features.t()

        logits_per_face /= self.temperature
        logits_per_ocular /= self.temperature
        logits_clstr_face /= self.temperature
        logits_clstr_ocular /= self.temperature

        positive_mask = self._get_positive_mask(face_features.shape[0])
        negatives_face = logits_clstr_face * positive_mask
        negatives_ocular = logits_clstr_ocular * positive_mask

        face_logits = torch.cat([logits_per_face, self.negative_w * negatives_face], dim=1)
        ocular_logits = torch.cat([logits_per_ocular, self.negative_w * negatives_ocular], dim=1)

        diag = np.eye(batch_size)
        mask_face = torch.from_numpy(diag).to(self.config.device)
        mask_ocular = torch.from_numpy(diag).to(self.config.device)

        mask_neg_f = torch.zeros_like(negatives_face)
        mask_neg_o = torch.zeros_like(negatives_ocular)
        mask_f = torch.cat([mask_face, mask_neg_f], dim=1)
        mask_o = torch.cat([mask_ocular, mask_neg_o], dim=1)

        loss_f = self.compute_loss(face_logits, mask_f)
        loss_o = self.compute_loss(ocular_logits, mask_o)

        return (loss_f.mean() + loss_o.mean()) / 2
