import torch
import torch.nn as nn
import torch.nn.functional as F
from det3d.core.utils.center_utils import _transpose_and_gather_feat

class RegLoss(nn.Module):
  '''Regression loss for an output tensor
    Arguments:
      output (batch x dim x h x w)
      mask (batch x max_objects)
      ind (batch x max_objects)
      target (batch x max_objects x dim)
  '''
  def __init__(self):
    super(RegLoss, self).__init__()
  
  def forward(self, output, mask, ind, target, det_scores=None, semi=False):
    if semi:
      pred = _transpose_and_gather_feat(output, ind)
      mask = mask.float().unsqueeze(2)
      det_scores = det_scores.float().unsqueeze(0) 

      loss = F.l1_loss(det_scores*pred*mask, target*mask, reduction='none')
      loss = loss / (mask.sum() + 1e-4)
      loss = loss.transpose(2 ,0).sum(dim=2).sum(dim=1)
    else:
      pred = _transpose_and_gather_feat(output, ind)
      mask = mask.float().unsqueeze(2) 

      loss = F.l1_loss(pred*mask, target*mask, reduction='none')
      loss = loss / (mask.sum() + 1e-4)
      loss = loss.transpose(2 ,0).sum(dim=2).sum(dim=1)
    return loss

class FastFocalLoss(nn.Module):
  '''
  Reimplemented focal loss, exactly the same as the CornerNet version.
  Faster and costs much less memory.
  '''
  def __init__(self):
    super(FastFocalLoss, self).__init__()

  def forward(self, out, target, ind, mask, cat, det_scores=None, alpha=0.0, thresh=0.5, semi=False):
    '''
    Arguments:
      out, target: B x C x H x W
      ind, mask: B x M
      cat (category id for peaks): B x M
    '''
    if semi:
      mask = mask.float()
      positive_mask = det_scores > 0.6
      gt_mask = det_scores.eq(1).float() * mask.unsqueeze(2)
      unlabel_mask = det_scores.lt(1).float() * mask.unsqueeze(2) * positive_mask.float()

      gt = torch.pow(1 - target, 4)
      neg_loss = torch.log(1 - out) * torch.pow(out, 2) * gt
      neg_loss = neg_loss.sum()

      pos_pred_pix = _transpose_and_gather_feat(out, ind) # B x M x C
      pos_pred = pos_pred_pix.gather(2, cat.unsqueeze(2)) # B x M
      num_pos = gt_mask.sum()
      pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2) * \
                gt_mask.unsqueeze(0)
      pos_loss = pos_loss.sum()

      num_semi = unlabel_mask.sum()
      det_scores = det_scores * unlabel_mask
      pos_semi_loss = torch.log(pos_pred) * det_scores.unsqueeze(0)

      # if num_semi > 0:
      #   import pdb; pdb.set_trace()
      pos_semi_loss = pos_semi_loss.sum()

      if num_pos == 0:
        return - neg_loss
      if num_pos > 0 and num_semi == 0:
        return - (pos_loss + neg_loss) / num_pos
      if num_pos == 0 and num_semi > 0:
        return - (pos_semi_loss + neg_loss) / num_semi
      return - (neg_loss / (num_semi + num_pos) + (pos_loss / num_pos) + (pos_semi_loss / num_semi))
    # if semi:
    #   # mask = mask.float()
    #   # # gt = torch.pow(1 - target, 4)
    #   # # neg_loss = torch.log(1 - out) * torch.pow(out, 2) * gt
    #   # # neg_loss = neg_loss.sum()

    #   # pos_pred_pix = _transpose_and_gather_feat(out, ind) # B x M x C
    #   # pos_pred = pos_pred_pix.gather(2, cat.unsqueeze(2)) # B x M
    #   # # num_pos = mask.sum()
    #   # # pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2) * torch.pow(det_scores.unsqueeze(0), \
    #   # #                   alpha) * mask.unsqueeze(2)
    #   # # pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2) * torch.pow(det_scores.unsqueeze(0), \
    #   # #                   alpha) * mask.unsqueeze(2)
    #   # loss = F.l1_loss(det_scores.unsqueeze(0), pos_pred * mask.unsqueeze(2), reduction='none')
    #   # loss = loss.sum()
    #   # return loss / (mask.sum() + 1e-4)
    #   # if num_pos == 0:
    #   #   return - neg_loss
    #   # return - (pos_loss + neg_loss) / num_pos
    #   mask = mask.float()
    #   negative_mask = det_scores < thresh
    #   positive_mask = ~negative_mask
    #   # negative_mask = (negative_mask.float() * mask.view(-1, 1))
    #   # positive_mask = (positive_mask.float() * mask.view(-1, 1))
    #   # import pdb; pdb.set_trace()
    #   # negative_mask = negative_mask.long()
    #   # positive_mask = positive_mask.long()
    #   gt_mask = det_scores.eq(1).float() * mask.unsqueeze(2)
    #   unlabel_mask = det_scores.lt(1).float() * mask.unsqueeze(2)

    #   gt = torch.pow(1 - target, 4)
    #   neg_loss = torch.log(1 - out) * torch.pow(out, 2) * gt
    #   neg_loss = neg_loss.sum()

    #   pos_pred_pix = _transpose_and_gather_feat(out, ind) # B x M x C
    #   pos_pred = pos_pred_pix.gather(2, cat.unsqueeze(2)) # B x M
    #   num_pos = gt_mask.sum()
    #   pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2) * \
    #             gt_mask.unsqueeze(0)
    #   pos_loss = pos_loss.sum()

    #   num_semi = unlabel_mask.sum()
    #   pos_pred_semi = pos_pred * positive_mask.float().unsqueeze(0) + \
    #                   (1. - pos_pred) * negative_mask.float().unsqueeze(0)

    #   det_scores[negative_mask] = (1. - det_scores)[negative_mask]
    #   # det_scores_clamp = torch.clamp(det_scores, min=1e-7, max =1. - 1e-7)
    #   # det_entropy = 1. + det_scores_clamp * torch.log(det_scores_clamp)
    #   det_entropy = 1. + det_scores * torch.log(det_scores)
    #   pos_semi_loss = torch.log(pos_pred_semi) * torch.pow(det_entropy.unsqueeze(0), \
    #                     alpha) * unlabel_mask.unsqueeze(0)
    #   # if num_semi > 0:
    #   #   import pdb; pdb.set_trace()
    #   pos_semi_loss = pos_semi_loss.sum()

    #   if num_pos == 0:
    #     return - neg_loss
    #   if num_pos > 0 and num_semi == 0:
    #     return - (pos_loss + neg_loss) / num_pos
    #   if num_pos == 0 and num_semi > 0:
    #     return - (pos_semi_loss + neg_loss) / num_semi
    #   return - (neg_loss / (num_semi + num_pos) + (pos_loss / num_pos) + (pos_semi_loss / num_semi))
    else:
      mask = mask.float()
      gt = torch.pow(1 - target, 4)
      neg_loss = torch.log(1 - out) * torch.pow(out, 2) * gt
      neg_loss = neg_loss.sum()

      pos_pred_pix = _transpose_and_gather_feat(out, ind) # B x M x C
      pos_pred = pos_pred_pix.gather(2, cat.unsqueeze(2)) # B x M
      num_pos = mask.sum()
      pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2) * \
                 mask.unsqueeze(2)
      pos_loss = pos_loss.sum()
      if num_pos == 0:
        return - neg_loss
      return - (pos_loss + neg_loss) / num_pos
