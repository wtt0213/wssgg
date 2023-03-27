import torch
import clip
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.model.model_utils.model_base import BaseModel
from utils import op_utils
from src.utils.eva_utils import get_gt, evaluate_topk_object, evaluate_topk_predicate, evaluate_topk, evaluate_triplet_topk
from src.model.model_utils.network_GNN import GraphEdgeAttenNetworkLayers
from src.model.model_utils.network_PointNet import PointNetfeat, PointNetCls, PointNetRelCls, PointNetRelClsMulti
from clip_adapter.model import AdapterModel

class OnlyMimic(BaseModel):
    def __init__(self, config, num_obj_class, num_rel_class, dim_descriptor=11):
        super().__init__('Mmgnet', config)

        self.mconfig = mconfig = config.MODEL
        with_bn = mconfig.WITH_BN

        dim_point = 3
        if mconfig.USE_RGB:
            dim_point +=3
        if mconfig.USE_NORMAL:
            dim_point +=3
        
        dim_f_spatial = dim_descriptor
        dim_point_rel = dim_f_spatial

        self.dim_point=dim_point
        self.dim_edge=dim_point_rel
        self.num_class=num_obj_class
        self.num_rel=num_rel_class
        self.flow = 'target_to_source'
        self.clip_feat_dim = self.config.MODEL.clip_feat_dim

        dim_point_feature = 768
        
        
        # Object Encoder
        self.obj_encoder = PointNetfeat(
            global_feat=True, 
            batch_norm=with_bn,
            point_size=dim_point, 
            input_transform=False,
            feature_transform=mconfig.feature_transform,
            out_size=dim_point_feature)      
        
        # Relationship Encoder
        self.rel_encoder = PointNetfeat(
            global_feat=True,
            batch_norm=with_bn,
            point_size=dim_point_rel,
            input_transform=False,
            feature_transform=mconfig.feature_transform,
            out_size=mconfig.edge_feature_size)
        
        self.gcn = GraphEdgeAttenNetworkLayers(512,
                            256,
                            self.mconfig.DIM_ATTEN,
                            self.mconfig.N_LAYERS, 
                            self.mconfig.NUM_HEADS,
                            self.mconfig.GCN_AGGR,
                            flow=self.flow,
                            attention=self.mconfig.ATTENTION,
                            use_edge=self.mconfig.USE_GCN_EDGE,
                            DROP_OUT_ATTEN=self.mconfig.DROP_OUT_ATTEN)

        # object adapter
        self.clip_adapter = AdapterModel(input_size=512, output_size=512, alpha=0.5)
        self.obj_logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # self.mlp = torch.nn.Sequential(
        #     torch.nn.Linear(512 + 256, 512),
        #     torch.nn.LayerNorm(512)
        # )
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(512 + 256, 512 - 8),
            torch.nn.BatchNorm1d(512 - 8),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1)
        )

        if mconfig.multi_rel_outputs:
            self.rel_predictor = PointNetRelClsMulti(
                num_rel_class, 
                in_size=mconfig.edge_feature_size, 
                batch_norm=with_bn,drop_out=True)
        else:
            self.rel_predictor = PointNetRelCls(
                num_rel_class, 
                in_size=mconfig.edge_feature_size, 
                batch_norm=with_bn,drop_out=True)

        self.init_weight(label_path='/data/caidaigang/project/3DSSG_Repo/data/3DSSG_subset/classes.txt', \
        adapter_path='/data/caidaigang/project/3DSSG_Repo/clip_adapter/checkpoint/origin_mean.pth')
        
        gcn_no_rel, gcn_rel = [], []
        for name, para in self.gcn.named_parameters():
            if 'nn_edge' in name:
                gcn_rel.append(para)
            else:
                gcn_no_rel.append(para)
        
        print(f"gcn_no_rel : \n{gcn_no_rel}")
        print(f"gcn_rel : \n{gcn_rel}")
        
        self.optimizer = optim.AdamW([
            {'params':self.obj_encoder.parameters(), 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params':self.rel_encoder.parameters(), 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params':gcn_no_rel, 'lr':float(config.LR) / 2, 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params':gcn_rel, 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params':self.obj_predictor.parameters(), 'lr':float(config.LR) / 10, 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params':self.rel_predictor.parameters(), 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            #{'params':self.obj_logit_scale, 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params':self.mlp.parameters(), 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            #{'params':self.clip_adapter.parameters(), 'lr':float(config.LR) / 10, 'weight_decay':1e-5, 'amsgrad':self.config.AMSGRAD},
        ])
        self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=self.config.max_iteration, last_epoch=-1)
        self.optimizer.zero_grad()

    def init_weight(self, label_path, adapter_path):
        torch.nn.init.xavier_uniform_(self.mlp[0].weight)
        text_features = self.get_label_weight(label_path)
        # node feature classifier        
        self.obj_predictor = torch.nn.Linear(self.mconfig.clip_feat_dim, self.num_class)
        self.obj_predictor.weight.data.copy_(text_features)
        for param in self.obj_predictor.parameters():
            param.requires_grad = True
        # self.obj_logit_scale = self.clip_adapter.obj_logit_scale
        # load adapter weights
        self.clip_adapter.load_state_dict(torch.load(adapter_path, 'cpu'))
        # freeze clip adapter
        for param in self.clip_adapter.parameters():
            param.requires_grad = False
        # self.obj_logit_scale.requires_grad = True
    
    def get_label_weight(self, label_path):
        label_list = []
        model, preprocess = clip.load("ViT-B/32", device='cuda')
        with open(label_path, "r") as f:
            data = f.readlines()
        for line in data:
            label_list.append(line.strip())
        # get norm clip weight
        text = torch.cat([clip.tokenize(f"a photo of a {c}") for c in label_list]).cuda()
        with torch.no_grad():
            text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features.float()

    def smooth_loss(self, pred, gold):
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
        return loss

    def forward(self, obj_points, obj_2d_feats, edge_indices, descriptor=None, batch_ids=None, istrain=False):

        obj_feature = self.obj_encoder(obj_points)
        obj_feature_tmp = obj_feature[..., :512].clone()

        obj_feature = self.mlp(obj_feature)

        if self.mconfig.USE_SPATIAL:
            tmp = descriptor[:,3:].clone()
            tmp[:,6:] = tmp[:,6:].log() # only log on volume and length
            obj_feature = torch.cat([obj_feature, tmp],dim=1)
        
        ''' Create edge feature '''
        with torch.no_grad():
            edge_feature = op_utils.Gen_edge_descriptor(flow=self.flow)(descriptor, edge_indices)
        
        rel_feature = self.rel_encoder(edge_feature)
        
        gcn_obj_feature, gcn_rel_feature, probs = self.gcn(obj_feature, rel_feature, edge_indices)

        rel_cls = self.rel_predictor(gcn_rel_feature)

        ''' Create 2d feature'''
        if istrain:
            with torch.no_grad():
                obj_2d_feats_finetune = self.clip_adapter(obj_2d_feats)

        logit_scale = self.obj_logit_scale.exp()

        obj_logits = logit_scale * self.obj_predictor(gcn_obj_feature / gcn_obj_feature.norm(dim=-1, keepdim=True))

        if istrain:
            return obj_logits, rel_cls, obj_feature_tmp, obj_2d_feats_finetune
        else:
            return obj_logits, rel_cls

    def process_train(self, obj_points, obj_2d_feats, gt_cls, descriptor, gt_rel_cls, edge_indices, batch_ids=None, with_log=False, ignore_none_rel=False, weights_obj=None, weights_rel=None):
        self.iteration +=1    
        
        obj_pred, rel_pred, obj_feature_3d, obj_feature_2d = self(obj_points, obj_2d_feats, edge_indices.t().contiguous(),descriptor, istrain=True)
        
        # compute loss for obj
        loss_obj = F.cross_entropy(obj_pred, gt_cls)

         # compute loss for rel
        if self.mconfig.multi_rel_outputs:
            if self.mconfig.WEIGHT_EDGE == 'BG':
                if self.mconfig.w_bg != 0:
                    weight = self.mconfig.w_bg * (1 - gt_rel_cls) + (1 - self.mconfig.w_bg) * gt_rel_cls
                else:
                    weight = None
            elif self.mconfig.WEIGHT_EDGE == 'DYNAMIC':
                batch_mean = torch.sum(gt_rel_cls, dim=(0))
                zeros = (gt_rel_cls.sum(-1) ==0).sum().unsqueeze(0)
                batch_mean = torch.cat([zeros,batch_mean],dim=0)
                weight = torch.abs(1.0 / (torch.log(batch_mean+1)+1)) # +1 to prevent 1 /log(1) = inf                
                if ignore_none_rel:
                    weight[0] = 0
                    weight *= 1e-2 # reduce the weight from ScanNet
                    # print('set weight of none to 0')
                if 'NONE_RATIO' in self.mconfig:
                    weight[0] *= self.mconfig.NONE_RATIO
                    
                weight[torch.where(weight==0)] = weight[0].clone() if not ignore_none_rel else 0# * 1e-3
                weight = weight[1:]                
            elif self.mconfig.WEIGHT_EDGE == 'OCCU':
                weight = weights_rel
            elif self.mconfig.WEIGHT_EDGE == 'NONE':
                weight = None
            else:
                raise NotImplementedError("unknown weight_edge type")
            loss_rel = F.binary_cross_entropy(rel_pred, gt_rel_cls, weight=weight)
        else:
            if self.mconfig.WEIGHT_EDGE == 'DYNAMIC':
                one_hot_gt_rel = torch.nn.functional.one_hot(gt_rel_cls,num_classes = self.num_rel)
                batch_mean = torch.sum(one_hot_gt_rel, dim=(0), dtype=torch.float)
                weight = torch.abs(1.0 / (torch.log(batch_mean+1)+1)) # +1 to prevent 1 /log(1) = inf
                if ignore_none_rel: 
                    weight[0] = 0 # assume none is the first relationship
                    weight *= 1e-2 # reduce the weight from ScanNet
            elif self.mconfig.WEIGHT_EDGE == 'OCCU':
                weight = weights_rel
            elif self.mconfig.WEIGHT_EDGE == 'BG':
                if self.mconfig.w_bg != 0:
                    weight = self.mconfig.w_bg * (1 - gt_rel_cls) + (1 - self.mconfig.w_bg) * gt_rel_cls
                else:
                    weight = None
            elif self.mconfig.WEIGHT_EDGE == 'NONE':
                weight = None
            else:
                raise NotImplementedError("unknown weight_edge type")

            if 'ignore_entirely' in self.mconfig and (self.mconfig.ignore_entirely and ignore_none_rel):
                loss_rel = torch.zeros(1,device=rel_pred.device, requires_grad=False)
            else:
                loss_rel = F.nll_loss(rel_pred, gt_rel_cls, weight = weight)
        
        lambda_r = 1.0
        lambda_o = self.mconfig.lambda_o
        lambda_max = max(lambda_r,lambda_o)
        lambda_r /= lambda_max
        lambda_o /= lambda_max

        # loss_mimic = F.mse_loss(obj_feature_3d, obj_feature_2d, reduction='sum')
        # loss_mimic /= obj_feature_3d.shape[0]
        obj_feature_3d = obj_feature_3d / obj_feature_3d.norm(dim=-1, keepdim=True)
        obj_feature_2d = obj_feature_2d / obj_feature_2d.norm(dim=-1, keepdim=True)
        loss_mimic = F.l1_loss(obj_feature_3d, obj_feature_2d)
        
        loss = lambda_o * loss_obj + 3 * lambda_r * loss_rel + 0.1 * loss_mimic
        self.backward(loss)
        
        # compute metric
        top_k_obj = evaluate_topk_object(obj_pred.detach(), gt_cls, topk=11)
        gt_edges = get_gt(gt_cls, gt_rel_cls, edge_indices, self.mconfig.multi_rel_outputs)
        top_k_rel = evaluate_topk_predicate(rel_pred.detach(), gt_edges, self.mconfig.multi_rel_outputs, topk=6)
        
        if not with_log:
            return top_k_obj, top_k_rel, loss_rel.detach(), loss_obj.detach(), loss.detach()

        obj_topk_list = [100 * (top_k_obj <= i).sum() / len(top_k_obj) for i in [1, 5, 10]]
        rel_topk_list = [100 * (top_k_rel <= i).sum() / len(top_k_rel) for i in [1, 3, 5]]
        
        
        log = [("train/rel_loss", loss_rel.detach().item()),
                ("train/obj_loss", loss_obj.detach().item()),
                ("train/mimic_loss", loss_mimic.detach().item()),
                ("train/loss", loss.detach().item()),
                ("train/Obj_R1", obj_topk_list[0]),
                ("train/Obj_R5", obj_topk_list[1]),
                ("train/Obj_R10", obj_topk_list[2]),
                ("train/Pred_R1", rel_topk_list[0]),
                ("train/Pred_R3", rel_topk_list[1]),
                ("train/Pred_R5", rel_topk_list[2]),
            ]
        return log
           
    def process_val(self, obj_points, obj_2d_feats, gt_cls, descriptor, gt_rel_cls, edge_indices, batch_ids=None, with_log=False, use_triplet=False):
 
        obj_pred, rel_pred = self(obj_points, None, edge_indices.t().contiguous(), descriptor, istrain=False)
        
        # compute metric
        top_k_obj = evaluate_topk_object(obj_pred.detach().cpu(), gt_cls, topk=11)
        gt_edges = get_gt(gt_cls, gt_rel_cls, edge_indices, self.mconfig.multi_rel_outputs)
        top_k_rel = evaluate_topk_predicate(rel_pred.detach().cpu(), gt_edges, self.mconfig.multi_rel_outputs, topk=6)
        if use_triplet:
            top_k_triplet, cls_matrix, sub_scores, obj_scores, rel_scores = evaluate_triplet_topk(obj_pred.detach().cpu(), rel_pred.detach().cpu(), gt_edges, edge_indices, self.mconfig.multi_rel_outputs, topk=101, use_clip=True, obj_topk=top_k_obj)
        else:
            top_k_triplet = [101]
            cls_matrix = None
            sub_scores = None
            obj_scores = None
            rel_scores = None

        return top_k_obj, top_k_obj, top_k_rel, top_k_rel, top_k_triplet, top_k_triplet, cls_matrix, sub_scores, obj_scores, rel_scores
 
    def backward(self, loss):
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        # update lr
        self.lr_scheduler.step()
