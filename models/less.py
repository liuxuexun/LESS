import functools
import pointgroup_ops
import spconv.pytorch as spconv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.backbone import ResidualBlock, UBlockPWCA
from transformers import RobertaModel, RobertaTokenizerFast

class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model=256, nhead=8, dropout=0.0):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, source, query, attn_masks=None, pe=None):
        """
        source (B*N, d_model)
        query Tensor (b, n_q, d_model)
        """
        query = self.with_pos_embed(query, pe)
        k = v = source
        if attn_masks:
            attn_masks = torch.stack(attn_masks, dim=0)
            output, _ = self.attn(query, k, v, attn_mask=attn_masks)  # (1, 100, d_model)
        else:
            output, _ = self.attn(query, k, v)
        self.dropout(output)
        output = output + query
        self.norm(output)
        return output


class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model=256, nhead=8, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, x, pe=None):
        """
        x Tensor (b, 100, c)
        """
        q = k = self.with_pos_embed(x, pe)
        output, _ = self.attn(q, k, x)
        output = self.dropout(output) + x
        output = self.norm(output)
        return output


class FFN(nn.Module):

    def __init__(self, d_model, hidden_dim, dropout=0.0, activation_fn='relu'):
        super().__init__()
        if activation_fn == 'relu':
            self.net = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, d_model),
                nn.Dropout(dropout),
            )
        elif activation_fn == 'gelu':
            self.net = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, d_model),
                nn.Dropout(dropout),
            )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        output = self.net(x)
        output = output + x
        output = self.norm(output)
        return output


class QMP(nn.Module):
    """
    in_channels List[int] (4,) [64,96,128,160]
    """

    def __init__(
        self,
        num_layer=6,
        num_query=100,
        in_channel=32,
        d_model=256,
        nhead=8,
        hidden_dim=1024,
        dropout=0.0,
        activation_fn='relu',
        attn_mask=False,
        pe=False,
    ):
        super().__init__()
        self.num_layer = num_layer
        self.num_query = num_query
        self.input_proj = nn.Sequential(nn.Linear(in_channel, d_model), nn.LayerNorm(d_model), nn.ReLU())
        self.query = nn.Embedding(num_query, d_model)
        if pe:
            self.pe = nn.Embedding(num_query, d_model)          
        self.cross_attn_layers = nn.ModuleList([])
        self.self_attn_layers = nn.ModuleList([])
        self.ffn_layers = nn.ModuleList([])
        for i in range(num_layer):
            self.cross_attn_layers.append(CrossAttentionLayer(d_model, nhead, dropout))
            self.self_attn_layers.append(SelfAttentionLayer(d_model, nhead, dropout))
            self.ffn_layers.append(FFN(d_model, hidden_dim, dropout, activation_fn))
        self.out_norm = nn.LayerNorm(d_model)
        self.out_cls = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 2))
        self.out_score = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))
        self.x_mask = nn.Sequential(nn.Linear(in_channel, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.attn_mask = attn_mask

    def get_mask(self, query, mask_feats, batch_offsets):
        pred_masks = []
        attn_masks = []
        for i in range(len(batch_offsets) - 1):
            pred_mask = torch.einsum('nd,md->nm', query[i], mask_feats[i])
            if self.attn_mask: 
                attn_mask = (pred_mask.sigmoid() < 0.5).bool()
                attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                attn_mask = attn_mask.detach()
                attn_masks.append(attn_mask)
            pred_masks.append(pred_mask)

        return pred_masks, attn_masks

    def forward(self, x, batch_offsets):
        inst_feats = self.input_proj(x)
        mask_feats = self.x_mask(x)
        B = len(batch_offsets) - 1
        query = self.query.weight.unsqueeze(0).repeat(B, 1, 1)  # (b, n, d_model)
        query = self.out_norm(query)
        pred_masks, att_masks = self.get_mask(query, mask_feats, batch_offsets)
        for i in range(self.num_layer):
            query = self.cross_attn_layers[i](inst_feats, query, att_masks)
            query = self.self_attn_layers[i](query)
            query = self.ffn_layers[i](query)
            query = self.out_norm(query)
            pred_masks, att_masks = self.get_mask(query, mask_feats, batch_offsets)

        pred_masks = torch.cat([pred_mask.unsqueeze(0) for pred_mask in pred_masks])
        
        return pred_masks, att_masks, query

class Unet(nn.Module):

    def __init__(
        self,
        input_channel: int = 6,
        blocks: int = 5,
        block_reps: int = 2,
        media: int = 32,
        normalize_before=True,
        return_blocks=True,
        fix_module=[],
        cfg=None
    ):
        super().__init__()

        self.cfg = cfg
        self.encoder_layer_num = 3
        self.inner_dim = 768
        self.dropout_rate = 0.15
        self.tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
        
        
        # backbone and pooling
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(
                input_channel,
                media,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key='subm1',
            ))
        block = ResidualBlock
        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)
        self.block_list = [media * (i + 1) for i in range(blocks)]
        self.num_heads_list = [1, 1, 1, 1, 1]
        self.unet = UBlockPWCA(
            self.block_list,
            self.num_heads_list,
            norm_fn,
            block_reps,
            block,
            indice_key_id=1,
            normalize_before=normalize_before,
            return_blocks=return_blocks,
        )
        
        self.output_layer = spconv.SparseSequential(norm_fn(media), nn.ReLU(inplace=True))
        
        self.language_encoder = RobertaModel.from_pretrained('roberta-base')
        
        self.word_fuse_layer = {}
        
        for dim in self.block_list:
            self.word_fuse_layer[f'word_fuse_layer_{dim}'] = \
            nn.Sequential(nn.Linear(self.inner_dim, self.inner_dim), 
            nn.ReLU(), nn.Dropout(self.dropout_rate), 
            nn.Linear(self.inner_dim, dim)).cuda()

        self.query_decoder = QMP(num_layer=1, nhead=1, num_query=20, attn_mask=True)
        self.sentense_output_layer = nn.Sequential(nn.Linear(self.inner_dim, self.inner_dim), 
            nn.ReLU(), nn.Dropout(self.dropout_rate), 
            nn.Linear(self.inner_dim, 256))
        
        # freeze
        for module in fix_module:
            module = getattr(self, module)
            module.eval()
            for param in module.parameters():
                param.requires_grad = False


    def forward(self, data_dict):
        
        # Text enconder
        word_feature = self.language_encoder(**data_dict['lang_tokens'])[0]
        sentence_feature = word_feature[:, 0, :]
        
        batch_size = word_feature.shape[0]
        voxel_feats = pointgroup_ops.voxelization(data_dict['feats'], data_dict['v2p_map'])
        input = spconv.SparseConvTensor(voxel_feats, data_dict['voxel_coords'].int(), data_dict['spatial_shape'], batch_size)
        
        # backbone + PWCA
        fuse_feats = self.extract_feat(input, data_dict['p2v_map'], word_feature[:, 1:, :], data_dict['lang_tokens'].data['attention_mask'][:, 1:])     

        # reshape [b,n,c]
        fuse_feats = fuse_feats.reshape(batch_size, -1, 32)
        
        # QMP
        pred_masks, _, query = self.query_decoder(fuse_feats, data_dict['batch_offset'])

        # QSA
        query_text_similarity = F.softmax(torch.matmul(query, self.sentense_output_layer(sentence_feature).unsqueeze(-1)), dim=1)
        
        data_dict['logits'] = torch.matmul(torch.permute(query_text_similarity, (0, 2, 1)), pred_masks).squeeze(1)
         
        data_dict['final_fea'] = fuse_feats

        return data_dict

    def extract_feat(self, x, v2p_map, word_feature, word_mask):
        x = self.input_conv(x)
        x, _ = self.unet(x, None, word_feature, self.word_fuse_layer, word_mask)
        x = self.output_layer(x)
        x = x.features[v2p_map.long()]  # (B*N, media)
        return x

def l2_norm(tensor, eps=1e-12):
    norm = torch.norm(tensor, p=2, dim=-1, keepdim=True)
    normalized_tensor = tensor / (norm + eps)
    return normalized_tensor

def Compute_loss(end_points):

    gt_mask = end_points['instance_refer_mask']
    pred_logits = end_points['logits']
    pred_logits = torch.sigmoid(pred_logits)
    
    # Segmentation Loss
    ce_label_weight = torch.zeros_like(gt_mask).float().cuda()
    ce_label_weight = torch.fill_(ce_label_weight, 1)
    ce_label_weight[gt_mask > 0.1] = 20
    criterion = nn.BCELoss(weight=ce_label_weight, reduction='none')
    loss_mask = criterion(pred_logits, gt_mask.float().cuda()).mean()
    
    # Area Regularization Loss
    loss_area = (torch.sum(pred_logits, dim=1) / pred_logits.shape[1]).mean() 
    
    # Point-to-Point Contrastive Loss
    temperature = 0.05
    loss_p2p = 0
    batchsize = end_points['final_fea'].shape[0]
    for i in range(batchsize):
        # L2 norm before p2p loss
        fea = l2_norm(end_points['final_fea'][i])
        mask = gt_mask[i].bool()
        positive_index = torch.where(mask == True)[0]
        negative_index = torch.where(mask == False)[0]    
        positive_fea = fea[positive_index, :]
        negative_fea = fea[negative_index, :]
        positive_avg = torch.mean(positive_fea, dim=0, keepdim=True)   
        positive_score = torch.exp(torch.matmul(positive_fea, positive_avg.permute(1, 0)) / temperature)
        negative2positive = torch.sum(torch.exp(torch.matmul(positive_fea, negative_fea.permute(1, 0)) / temperature), dim=1, keepdim=True)
        loss_p2p += -torch.log(torch.mean((positive_score / (positive_score + negative2positive))))
    loss_p2p /= batchsize
    
    # Total Loss
    loss = loss_mask + loss_area + 0.05*loss_p2p

    end_points['loss'] = loss
    end_points['loss_mask'] = loss_mask
    end_points['loss_area'] = loss_area
    end_points['loss_p2p'] = 0.05*loss_p2p
    
    return loss, end_points


def Cal_index_IOU(mask_index1, mask_index2):
    inter_count = np.intersect1d(mask_index1, mask_index2).shape[0]
    union_count = np.union1d(mask_index1, mask_index2).shape[0]
    iou = inter_count / union_count
    return iou, inter_count, union_count

def Compute_iou(end_points):
    
    mask_ious = []
    inter_counts = []
    union_counts = []
    
    for i in range(end_points['instance_refer_mask'].shape[0]):
        pred_logits = end_points['logits'][i]
        pred_mask = torch.sigmoid(pred_logits).detach().cpu().numpy() > 0.5

        # A simple trick
        if np.sum(pred_mask) < 100:
            index = np.argsort(torch.sigmoid(pred_logits).detach().cpu().numpy())[-5000:]
            pred_mask[index] = True
            
        gt_mask = end_points['instance_refer_mask'][i]
    
        pred_mask_index = np.nonzero(pred_mask)[0]
        gt_mask_index = np.nonzero(gt_mask.detach().cpu().numpy())[0]
            
        mask_iou, I, U = Cal_index_IOU(pred_mask_index, gt_mask_index)
        mask_ious.append(mask_iou)
        inter_counts.append(I)
        union_counts.append(U)
        
    return np.mean(mask_ious), end_points, inter_counts, union_counts