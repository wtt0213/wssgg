if __name__ == '__main__' and __package__ is None:
    from os import sys
    sys.path.append('../')
import copy
import os

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from src.dataset.DataLoader import (CustomDataLoader, collate_fn_all,
                                    collate_fn_all_2d, collate_fn_all_des,
                                    collate_fn_det, collate_fn_mmg)
from src.dataset.dataset_builder import build_dataset
from src.model.SGFN_MMG.baseline_sgfn import Baseline
#from src.model.SGFN_MMG.baseline_sgpn import Baseline
from src.utils import op_utils
from src.utils.config import Config
from src.utils.eva_utils import get_mean_recall, get_zero_shot_recall


class MMGNet():
    def __init__(self, config):
        self.config = config
        self.model_name = self.config.NAME
        self.mconfig = mconfig = config.MODEL
        self.exp = config.exp
        self.save_res = config.EVAL
        self.update_2d = config.update_2d
        
        ''' Build dataset '''
        dataset = None
        if config.MODE  == 'train':
            if config.VERBOSE: print('build train dataset')
            self.dataset_train = build_dataset(self.config,split_type='train_scans', shuffle_objs=True,
                                               multi_rel_outputs=mconfig.multi_rel_outputs,
                                               use_rgb=mconfig.USE_RGB,
                                               use_normal=mconfig.USE_NORMAL)
            self.dataset_train.__getitem__(0)
                
        if config.MODE  == 'train' or config.MODE  == 'trace':
            if config.VERBOSE: print('build valid dataset')
            self.dataset_valid = build_dataset(self.config,split_type='validation_scans', shuffle_objs=False, 
                                      multi_rel_outputs=mconfig.multi_rel_outputs,
                                      use_rgb=mconfig.USE_RGB,
                                      use_normal=mconfig.USE_NORMAL)
            dataset = self.dataset_valid

        num_obj_class = len(self.dataset_valid.classNames)   
        num_rel_class = len(self.dataset_valid.relationNames)
        self.num_obj_class = num_obj_class
        self.num_rel_class = num_rel_class
        
        self.total = self.config.total = len(self.dataset_train) // self.config.Batch_Size
        self.max_iteration = self.config.max_iteration = int(float(self.config.MAX_EPOCHES)*len(self.dataset_train) // self.config.Batch_Size)
        self.max_iteration_scheduler = self.config.max_iteration_scheduler = int(float(100)*len(self.dataset_train) // self.config.Batch_Size)
        
        ''' Build Model '''
        self.model = Baseline(self.config, num_obj_class, num_rel_class).to(config.DEVICE)
        self.samples_path = os.path.join(config.PATH, self.model_name, self.exp,  'samples')
        self.results_path = os.path.join(config.PATH, self.model_name, self.exp, 'results')
        self.trace_path = os.path.join(config.PATH, self.model_name, self.exp, 'traced')
        self.writter = None
        
        if not self.config.EVAL:
            pth_log = os.path.join(config.PATH, "logs", self.model_name, self.exp)
            self.writter = SummaryWriter(pth_log)
        
    def load(self, best=False):
        return self.model.load(best)
        
    @torch.no_grad()
    def data_processing_train(self, items):
        obj_points, obj_2d_feats, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids = items 
        obj_points = obj_points.permute(0,2,1).contiguous()
        obj_points, obj_2d_feats, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids = \
            self.cuda(obj_points, obj_2d_feats, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids)
        return obj_points, obj_2d_feats, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids
    
    @torch.no_grad()
    def data_processing_val(self, items):
        obj_points, obj_2d_feats, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids = items 
        obj_points = obj_points.permute(0,2,1).contiguous()
        obj_points, obj_2d_feats, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids = \
            self.cuda(obj_points, obj_2d_feats, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids)
        return obj_points, obj_2d_feats, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids
          
    def train(self):
        ''' create data loader '''
        drop_last = True
        train_loader = CustomDataLoader(
            config = self.config,
            dataset=self.dataset_train,
            batch_size=self.config.Batch_Size,
            num_workers=self.config.WORKERS,
            drop_last=drop_last,
            shuffle=True,
            collate_fn=collate_fn_mmg,
        )
        
        self.model.epoch = 1
        keep_training = True
        
        if self.total == 1:
            print('No training data was provided! Check \'TRAIN_FLIST\' value in the configuration file.')
            return
        
        progbar = op_utils.Progbar(self.total, width=20, stateful_metrics=['Misc/epo', 'Misc/it', 'Misc/lr'])
                
        ''' Resume data loader to the last read location '''
        loader = iter(train_loader)
                   
        if self.mconfig.use_pretrain != "":
            self.model.load_pretrain_model(self.mconfig.use_pretrain, is_freeze=True)
        
        for k, p in self.model.named_parameters():
            if p.requires_grad:
                print(f"Para {k} need grad")
        ''' Train '''
        while(keep_training):

            if self.model.epoch > self.config.MAX_EPOCHES:
                break

            print('\n\nTraining epoch: %d' % self.model.epoch)
            
            for items in loader:
                self.model.train()
                
                ''' get data '''
                obj_points, obj_2d_feats, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids = self.data_processing_train(items)
                logs = self.model.process_train(obj_points, obj_2d_feats, gt_class, descriptor, gt_rel_cls, edge_indices, batch_ids, with_log=True,
                                                weights_obj=self.dataset_train.w_cls_obj, 
                                                weights_rel=self.dataset_train.w_cls_rel,
                                                ignore_none_rel = False)
                
                iteration = self.model.iteration
                logs += [
                    ("Misc/epo", int(self.model.epoch)),
                    ("Misc/it", int(iteration)),
                    ("lr", self.model.lr_scheduler.get_last_lr()[0])
                ]
                
                progbar.add(1, values=logs \
                            if self.config.VERBOSE else [x for x in logs if not x[0].startswith('Loss')])
                if self.config.LOG_INTERVAL and iteration % self.config.LOG_INTERVAL == 0:
                    self.log(logs, iteration)
                if self.model.iteration >= self.max_iteration:
                    break

            progbar = op_utils.Progbar(self.total, width=20, stateful_metrics=['Misc/epo', 'Misc/it'])
            loader = iter(train_loader)
            self.save()

            if ('VALID_INTERVAL' in self.config and self.config.VALID_INTERVAL > 0 and self.model.epoch % self.config.VALID_INTERVAL == 0):
                print('start validation...')
                rel_acc_val = self.validation()
                self.model.eva_res = rel_acc_val
                self.save()
            
            self.model.epoch += 1

            # if self.update_2d:
            #     print('load copy model from last epoch')
            #     # copy param from previous epoch
            #     model_pre = Mmgnet(self.config, self.num_obj_class, self.num_rel_class).to(self.config.DEVICE)
            #     for k, p in model_pre.named_parameters():
            #         p.data.copy_(self.model.state_dict()[k])
            #     model_pre.model_pre = None
            #     self.model.update_model_pre(model_pre)
                   
    def cuda(self, *args):
        return [item.to(self.config.DEVICE) for item in args]
    
    def log(self, logs, iteration):
        # Tensorboard
        if self.writter is not None and not self.config.EVAL:
            for i in logs:
                if not i[0].startswith('Misc'):
                    self.writter.add_scalar(i[0], i[1], iteration)
                    
    def save(self):
        self.model.save ()
        
    def validation(self, debug_mode = False):
        val_loader = CustomDataLoader(
            config = self.config,
            dataset=self.dataset_valid,
            batch_size=1,
            num_workers=self.config.WORKERS,
            drop_last=False,
            shuffle=False,
            collate_fn=collate_fn_mmg
        )
       
        total = len(self.dataset_valid)
        progbar = op_utils.Progbar(total, width=20, stateful_metrics=['Misc/it'])
        
        print('===   start evaluation   ===')
        self.model.eval()
        topk_obj_list, topk_rel_list, topk_triplet_list, cls_matrix_list, edge_feature_list = np.array([]), np.array([]), np.array([]), [], []
        sub_scores_list, obj_scores_list, rel_scores_list = [], [], []
        topk_obj_2d_list, topk_rel_2d_list, topk_triplet_2d_list = np.array([]), np.array([]), np.array([])

        for i, items in enumerate(val_loader, 0):
            ''' get data '''
            obj_points, obj_2d_feats, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids = self.data_processing_val(items)            
            
            with torch.no_grad():
                # if self.model.config.EVAL:
                #     top_k_obj, top_k_rel, tok_k_triplet, cls_matrix, sub_scores, obj_scores, rel_scores \
                #         = self.model.process_val(obj_points, gt_class, descriptor, gt_rel_cls, edge_indices, use_triplet=True)
                # else:
                top_k_obj, top_k_obj_2d, top_k_rel, top_k_rel_2d, tok_k_triplet, top_k_2d_triplet, cls_matrix, sub_scores, obj_scores, rel_scores \
                    = self.model.process_val(obj_points, obj_2d_feats, gt_class, descriptor, gt_rel_cls, edge_indices, batch_ids, use_triplet=True)
                        
            ''' calculate metrics '''
            topk_obj_list = np.concatenate((topk_obj_list, top_k_obj))
            topk_obj_2d_list = np.concatenate((topk_obj_2d_list, top_k_obj_2d))
            topk_rel_list = np.concatenate((topk_rel_list, top_k_rel))
            topk_rel_2d_list = np.concatenate((topk_rel_2d_list, top_k_rel_2d))
            topk_triplet_list = np.concatenate((topk_triplet_list, tok_k_triplet))
            topk_triplet_2d_list = np.concatenate((topk_triplet_2d_list, top_k_2d_triplet))
            if cls_matrix is not None:
                cls_matrix_list.extend(cls_matrix)
                sub_scores_list.extend(sub_scores)
                obj_scores_list.extend(obj_scores)
                rel_scores_list.extend(rel_scores)

            
            logs = [("Acc@1/obj_cls_acc", (topk_obj_list <= 1).sum() * 100 / len(topk_obj_list)),
                    ("Acc@1/obj_cls_2d_acc", (topk_obj_2d_list <= 1).sum() * 100 / len(topk_obj_2d_list)),
                    ("Acc@5/obj_cls_acc", (topk_obj_list <= 5).sum() * 100 / len(topk_obj_list)),
                    ("Acc@5/obj_cls_2d_acc", (topk_obj_2d_list <= 5).sum() * 100 / len(topk_obj_2d_list)),
                    ("Acc@10/obj_cls_acc", (topk_obj_list <= 10).sum() * 100 / len(topk_obj_list)),
                    ("Acc@10/obj_cls_2d_acc", (topk_obj_2d_list <= 10).sum() * 100 / len(topk_obj_2d_list)),
                    ("Acc@1/rel_cls_acc", (topk_rel_list <= 1).sum() * 100 / len(topk_rel_list)),
                    ("Acc@1/rel_cls_2d_acc", (topk_rel_2d_list <= 1).sum() * 100 / len(topk_rel_2d_list)),
                    ("Acc@3/rel_cls_acc", (topk_rel_list <= 3).sum() * 100 / len(topk_rel_list)),
                    ("Acc@3/rel_cls_2d_acc", (topk_rel_2d_list <= 3).sum() * 100 / len(topk_rel_2d_list)),
                    ("Acc@5/rel_cls_acc", (topk_rel_list <= 5).sum() * 100 / len(topk_rel_list)),
                    ("Acc@5/rel_cls_2d_acc", (topk_rel_2d_list <= 5).sum() * 100 / len(topk_rel_2d_list)),
                    ("Acc@50/triplet_acc", (topk_triplet_list <= 50).sum() * 100 / len(topk_triplet_list)),
                    ("Acc@50/triplet_2d_acc", (topk_triplet_2d_list <= 50).sum() * 100 / len(topk_triplet_2d_list)),
                    ("Acc@100/triplet_acc", (topk_triplet_list <= 100).sum() * 100 / len(topk_triplet_list)),
                    ("Acc@100/triplet_2d_acc", (topk_triplet_2d_list <= 100).sum() * 100 / len(topk_triplet_2d_list)),]

            progbar.add(1, values=logs if self.config.VERBOSE else [x for x in logs if not x[0].startswith('Loss')])


        cls_matrix_list = np.stack(cls_matrix_list)
        sub_scores_list = np.stack(sub_scores_list)
        obj_scores_list = np.stack(obj_scores_list)
        rel_scores_list = np.stack(rel_scores_list)
        mean_recall = get_mean_recall(topk_triplet_list, cls_matrix_list)
        mean_recall_2d = get_mean_recall(topk_triplet_2d_list, cls_matrix_list)
        zero_shot_recall, non_zero_shot_recall, all_zero_shot_recall = get_zero_shot_recall(topk_triplet_list, cls_matrix_list, self.dataset_valid.classNames, self.dataset_valid.relationNames)
        
        if self.model.config.EVAL:
            save_path = os.path.join(self.config.PATH, "results", self.model_name, self.exp)
            os.makedirs(save_path, exist_ok=True)
            np.save(os.path.join(save_path,'topk_pred_list.npy'), topk_rel_list )
            np.save(os.path.join(save_path,'topk_triplet_list.npy'), topk_triplet_list)
            np.save(os.path.join(save_path,'cls_matrix_list.npy'), cls_matrix_list)
            np.save(os.path.join(save_path,'sub_scores_list.npy'), sub_scores_list)
            np.save(os.path.join(save_path,'obj_scores_list.npy'), obj_scores_list)
            np.save(os.path.join(save_path,'rel_scores_list.npy'), rel_scores_list)
            f_in = open(os.path.join(save_path, 'result.txt'), 'w')
        else:
            f_in = None   
        
        obj_acc_1 = (topk_obj_list <= 1).sum() * 100 / len(topk_obj_list)
        obj_acc_2d_1 = (topk_obj_2d_list <= 1).sum() * 100 / len(topk_obj_2d_list)
        obj_acc_5 = (topk_obj_list <= 5).sum() * 100 / len(topk_obj_list)
        obj_acc_2d_5 = (topk_obj_2d_list <= 5).sum() * 100 / len(topk_obj_2d_list)
        obj_acc_10 = (topk_obj_list <= 10).sum() * 100 / len(topk_obj_list)
        obj_acc_2d_10 = (topk_obj_2d_list <= 10).sum() * 100 / len(topk_obj_2d_list)
        rel_acc_1 = (topk_rel_list <= 1).sum() * 100 / len(topk_rel_list)
        rel_acc_2d_1 = (topk_rel_2d_list <= 1).sum() * 100 / len(topk_rel_2d_list)
        rel_acc_3 = (topk_rel_list <= 3).sum() * 100 / len(topk_rel_list)
        rel_acc_2d_3 = (topk_rel_2d_list <= 3).sum() * 100 / len(topk_rel_2d_list)
        rel_acc_5 = (topk_rel_list <= 5).sum() * 100 / len(topk_rel_list)
        rel_acc_2d_5 = (topk_rel_2d_list <= 5).sum() * 100 / len(topk_rel_2d_list)
        triplet_acc_50 = (topk_triplet_list <= 50).sum() * 100 / len(topk_triplet_list)
        triplet_acc_2d_50 = (topk_triplet_2d_list <= 50).sum() * 100 / len(topk_triplet_2d_list)
        triplet_acc_100 = (topk_triplet_list <= 100).sum() * 100 / len(topk_triplet_list)
        triplet_acc_2d_100 = (topk_triplet_2d_list <= 100).sum() * 100 / len(topk_triplet_2d_list)

        rel_acc_mean_1, rel_acc_mean_3, rel_acc_mean_5 = self.compute_mean_predicate(cls_matrix_list, topk_rel_list)
        rel_acc_2d_mean_1, rel_acc_2d_mean_3, rel_acc_2d_mean_5 = self.compute_mean_predicate(cls_matrix_list, topk_rel_2d_list)

     
        print(f"Eval: 3d obj Acc@1  : {obj_acc_1}", file=f_in)   
        print(f"Eval: 2d obj Acc@1: {obj_acc_2d_1}", file=f_in)
        print(f"Eval: 3d obj Acc@5  : {obj_acc_5}", file=f_in) 
        print(f"Eval: 2d obj Acc@5: {obj_acc_2d_5}", file=f_in)  
        print(f"Eval: 3d obj Acc@10 : {obj_acc_10}", file=f_in)  
        print(f"Eval: 2d obj Acc@10: {obj_acc_2d_10}", file=f_in)
        print(f"Eval: 3d rel Acc@1  : {rel_acc_1}", file=f_in) 
        print(f"Eval: 3d mean rel Acc@1  : {rel_acc_mean_1}", file=f_in)   
        print(f"Eval: 2d rel Acc@1: {rel_acc_2d_1}", file=f_in)
        print(f"Eval: 2d mean rel Acc@1: {rel_acc_2d_mean_1}", file=f_in)
        print(f"Eval: 3d rel Acc@3  : {rel_acc_3}", file=f_in)   
        print(f"Eval: 3d mean rel Acc@3  : {rel_acc_mean_3}", file=f_in) 
        print(f"Eval: 2d rel Acc@3: {rel_acc_2d_3}", file=f_in)
        print(f"Eval: 2d mean rel Acc@3: {rel_acc_2d_mean_3}", file=f_in)
        print(f"Eval: 3d rel Acc@5  : {rel_acc_5}", file=f_in)
        print(f"Eval: 3d mean rel Acc@5  : {rel_acc_mean_5}", file=f_in) 
        print(f"Eval: 2d rel Acc@5: {rel_acc_2d_5}", file=f_in)
        print(f"Eval: 2d mean rel Acc@5: {rel_acc_2d_mean_5}", file=f_in)
        print(f"Eval: 3d triplet Acc@50 : {triplet_acc_50}", file=f_in)
        print(f"Eval: 2d triplet Acc@50: {triplet_acc_2d_50}", file=f_in)
        print(f"Eval: 3d triplet Acc@100 : {triplet_acc_100}", file=f_in)
        print(f"Eval: 2d triplet Acc@100: {triplet_acc_2d_100}", file=f_in)
        print(f"Eval: 3d mean recall@50 : {mean_recall[0]}", file=f_in)
        print(f"Eval: 2d mean recall@50: {mean_recall_2d[0]}", file=f_in)
        print(f"Eval: 3d mean recall@100 : {mean_recall[1]}", file=f_in)
        print(f"Eval: 2d mean recall@100: {mean_recall_2d[1]}", file=f_in)
        print(f"Eval: 3d zero-shot recall@50 : {zero_shot_recall[0]}", file=f_in)
        print(f"Eval: 3d zero-shot recall@100: {zero_shot_recall[1]}", file=f_in)
        print(f"Eval: 3d non-zero-shot recall@50 : {non_zero_shot_recall[0]}", file=f_in)
        print(f"Eval: 3d non-zero-shot recall@100: {non_zero_shot_recall[1]}", file=f_in)
        print(f"Eval: 3d all-zero-shot recall@50 : {all_zero_shot_recall[0]}", file=f_in)
        print(f"Eval: 3d all-zero-shot recall@100: {all_zero_shot_recall[1]}", file=f_in)



        if self.model.config.EVAL:
            f_in.close()
        
        logs = [("Acc@1/obj_cls_acc", obj_acc_1),
                ("Acc@1/obj_2d_cls_acc", obj_acc_2d_1),
                ("Acc@5/obj_cls_acc", obj_acc_5),
                ("Acc@5/obj_2d_cls_acc", obj_acc_2d_5),
                ("Acc@10/obj_cls_acc", obj_acc_10),
                ("Acc@10/obj_2d_cls_acc", obj_acc_2d_10),
                ("Acc@1/rel_cls_acc", rel_acc_1),
                ("Acc@1/rel_cls_acc_mean", rel_acc_mean_1),
                ("Acc@1/rel_2d_cls_acc", rel_acc_2d_1),
                ("Acc@1/rel_2d_cls_acc_mean", rel_acc_2d_mean_1),
                ("Acc@3/rel_cls_acc", rel_acc_3),
                ("Acc@3/rel_cls_acc_mean", rel_acc_mean_3),
                ("Acc@3/rel_2d_cls_acc", rel_acc_2d_3),
                ("Acc@3/rel_2d_cls_acc_mean", rel_acc_2d_mean_3),
                ("Acc@5/rel_cls_acc", rel_acc_5),
                ("Acc@5/rel_cls_acc_mean", rel_acc_mean_5),
                ("Acc@5/rel_2d_cls_acc", rel_acc_2d_5),
                ("Acc@5/rel_2d_cls_acc_mean", rel_acc_2d_mean_5),
                ("Acc@50/triplet_acc", triplet_acc_50),
                ("Acc@50/triplet_2d_acc", triplet_acc_2d_50),
                ("Acc@100/triplet_acc", triplet_acc_100),
                ("Acc@100/triplet_2d_acc", triplet_acc_2d_100),
                ("mean_recall@50", mean_recall[0]),
                ("mean_2d_recall@50", mean_recall_2d[0]),
                ("mean_recall@100", mean_recall[1]),
                ("mean_2d_recall@100", mean_recall_2d[1]),
                ("zero_shot_recall@50", zero_shot_recall[0]),
                ("zero_shot_recall@100", zero_shot_recall[1]),
                ("non_zero_shot_recall@50", non_zero_shot_recall[0]),
                ("non_zero_shot_recall@100", non_zero_shot_recall[1]),
                ("all_zero_shot_recall@50", all_zero_shot_recall[0]),
                ("all_zero_shot_recall@100", all_zero_shot_recall[1])
                ]
        
        self.log(logs, self.model.iteration)
        return mean_recall[0]

    def validation2(self, debug_mode = False):
        val_loader = CustomDataLoader(
            config = self.config,
            dataset=self.dataset_valid,
            batch_size=1,
            num_workers=self.config.WORKERS,
            drop_last=False,
            shuffle=False,
            collate_fn=collate_fn_mmg
        )
       
        total = len(self.dataset_valid)
        progbar = op_utils.Progbar(total, width=20, stateful_metrics=['Misc/it'])
        
        print('===   start evaluation   ===')
        self.model.eval()
        topk_obj_list, topk_rel_list, topk_triplet_list, cls_matrix_list, edge_feature_list = np.array([]), np.array([]), np.array([]), [], []
        sub_scores_list, obj_scores_list, rel_scores_list = [], [], []
        topk_obj_2d_list, topk_rel_2d_list, topk_triplet_2d_list = np.array([]), np.array([]), np.array([])
        recall_predcls_gc, recall_predcls_ngc, recall_sgcls_gc, recall_sgcls_ngc = [], [], [], []

        for i, items in enumerate(val_loader, 0):
            ''' get data '''
            obj_points, obj_2d_feats, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids = self.data_processing_val(items)            
            
            with torch.no_grad():
                # if self.model.config.EVAL:
                #     top_k_obj, top_k_rel, tok_k_triplet, cls_matrix, sub_scores, obj_scores, rel_scores \
                #         = self.model.process_val(obj_points, gt_class, descriptor, gt_rel_cls, edge_indices, use_triplet=True)
                # else:
                #top_k_obj, top_k_obj_2d, top_k_rel, top_k_rel_2d, tok_k_triplet, top_k_2d_triplet, cls_matrix, sub_scores, obj_scores, rel_scores \
                top_k_obj, top_k_obj_2d, top_k_rel, top_k_rel_2d, predcls_gc, predcls_ngc, sgcls_gc, sgcls_ngc  \
                    = self.model.process_val2(obj_points, obj_2d_feats, gt_class, descriptor, gt_rel_cls, edge_indices, batch_ids, use_triplet=True)
                        
            ''' calculate metrics '''
            topk_obj_list = np.concatenate((topk_obj_list, top_k_obj))
            topk_obj_2d_list = np.concatenate((topk_obj_2d_list, top_k_obj_2d))
            topk_rel_list = np.concatenate((topk_rel_list, top_k_rel))
            topk_rel_2d_list = np.concatenate((topk_rel_2d_list, top_k_rel_2d))
            #topk_triplet_list = np.concatenate((topk_triplet_list, tok_k_triplet))
            #topk_triplet_2d_list = np.concatenate((topk_triplet_2d_list, top_k_2d_triplet))
            # if cls_matrix is not None:
            #     cls_matrix_list.extend(cls_matrix)
            #     sub_scores_list.extend(sub_scores)
            #     obj_scores_list.extend(obj_scores)
            #     rel_scores_list.extend(rel_scores)
            recall_predcls_gc.append(predcls_gc)
            recall_predcls_ngc.append(predcls_ngc)
            recall_sgcls_gc.append(sgcls_gc)
            recall_sgcls_ngc.append(sgcls_ngc)

            
            logs = [("Acc@1/obj_cls_acc", (topk_obj_list <= 1).sum() * 100 / len(topk_obj_list)),
                    ("Acc@1/obj_cls_2d_acc", (topk_obj_2d_list <= 1).sum() * 100 / len(topk_obj_2d_list)),
                    ("Acc@5/obj_cls_acc", (topk_obj_list <= 5).sum() * 100 / len(topk_obj_list)),
                    ("Acc@5/obj_cls_2d_acc", (topk_obj_2d_list <= 5).sum() * 100 / len(topk_obj_2d_list)),
                    ("Acc@10/obj_cls_acc", (topk_obj_list <= 10).sum() * 100 / len(topk_obj_list)),
                    ("Acc@10/obj_cls_2d_acc", (topk_obj_2d_list <= 10).sum() * 100 / len(topk_obj_2d_list)),
                    ("Acc@1/rel_cls_acc", (topk_rel_list <= 1).sum() * 100 / len(topk_rel_list)),
                    ("Acc@1/rel_cls_2d_acc", (topk_rel_2d_list <= 1).sum() * 100 / len(topk_rel_2d_list)),
                    ("Acc@3/rel_cls_acc", (topk_rel_list <= 3).sum() * 100 / len(topk_rel_list)),
                    ("Acc@3/rel_cls_2d_acc", (topk_rel_2d_list <= 3).sum() * 100 / len(topk_rel_2d_list)),
                    ("Acc@5/rel_cls_acc", (topk_rel_list <= 5).sum() * 100 / len(topk_rel_list)),
                    ("Acc@5/rel_cls_2d_acc", (topk_rel_2d_list <= 5).sum() * 100 / len(topk_rel_2d_list)),
                    ("R20/predcls_gc", (np.stack(recall_predcls_gc).mean(0)[0])),
                    ("R50/predcls_gc", (np.stack(recall_predcls_gc).mean(0)[1])),
                    ("R100/predcls_gc", (np.stack(recall_predcls_gc).mean(0)[2])),
                    ("R20/predcls_ngc", (np.stack(recall_predcls_ngc).mean(0)[0])),
                    ("R50/predcls_ngc", (np.stack(recall_predcls_ngc).mean(0)[1])),
                    ("R100/predcls_ngc", (np.stack(recall_predcls_ngc).mean(0)[2])),
                    ("R20/sgcls_gc", (np.stack(recall_sgcls_gc).mean(0)[0])),
                    ("R50/sgcls_gc", (np.stack(recall_sgcls_gc).mean(0)[1])),
                    ("R100/sgcls_gc", (np.stack(recall_sgcls_gc).mean(0)[2])),
                    ("R20/sgcls_ngc", (np.stack(recall_sgcls_ngc).mean(0)[0])),
                    ("R50/sgcls_ngc", (np.stack(recall_sgcls_ngc).mean(0)[1])),
                    ("R100/sgcls_ngc", (np.stack(recall_sgcls_ngc).mean(0)[2]))]

                    #("Acc@50/triplet_acc", (topk_triplet_list <= 50).sum() * 100 / len(topk_triplet_list)),
                    #("Acc@50/triplet_2d_acc", (topk_triplet_2d_list <= 50).sum() * 100 / len(topk_triplet_2d_list)),
                    #("Acc@100/triplet_acc", (topk_triplet_list <= 100).sum() * 100 / len(topk_triplet_list)),
                    #("Acc@100/triplet_2d_acc", (topk_triplet_2d_list <= 100).sum() * 100 / len(topk_triplet_2d_list)),]

            progbar.add(1, values=logs if self.config.VERBOSE else [x for x in logs if not x[0].startswith('Loss')])


        # cls_matrix_list = np.stack(cls_matrix_list)
        # sub_scores_list = np.stack(sub_scores_list)
        # obj_scores_list = np.stack(obj_scores_list)
        # rel_scores_list = np.stack(rel_scores_list)
        # mean_recall = get_mean_recall(topk_triplet_list, cls_matrix_list)
        # mean_recall_2d = get_mean_recall(topk_triplet_2d_list, cls_matrix_list)
        # zero_shot_recall, non_zero_shot_recall, all_zero_shot_recall = get_zero_shot_recall(topk_triplet_list, cls_matrix_list, self.dataset_valid.classNames, self.dataset_valid.relationNames)
        
        if self.model.config.EVAL:
            save_path = os.path.join(self.config.PATH, "results", self.model_name, self.exp)
            os.makedirs(save_path, exist_ok=True)
            # np.save(os.path.join(save_path,'topk_pred_list.npy'), topk_rel_list )
            # np.save(os.path.join(save_path,'topk_triplet_list.npy'), topk_triplet_list)
            # np.save(os.path.join(save_path,'cls_matrix_list.npy'), cls_matrix_list)
            # np.save(os.path.join(save_path,'sub_scores_list.npy'), sub_scores_list)
            # np.save(os.path.join(save_path,'obj_scores_list.npy'), obj_scores_list)
            # np.save(os.path.join(save_path,'rel_scores_list.npy'), rel_scores_list)
            f_in = open(os.path.join(save_path, 'result2.txt'), 'w')
        else:
            f_in = None   
        
        obj_acc_1 = (topk_obj_list <= 1).sum() * 100 / len(topk_obj_list)
        obj_acc_2d_1 = (topk_obj_2d_list <= 1).sum() * 100 / len(topk_obj_2d_list)
        obj_acc_5 = (topk_obj_list <= 5).sum() * 100 / len(topk_obj_list)
        obj_acc_2d_5 = (topk_obj_2d_list <= 5).sum() * 100 / len(topk_obj_2d_list)
        obj_acc_10 = (topk_obj_list <= 10).sum() * 100 / len(topk_obj_list)
        obj_acc_2d_10 = (topk_obj_2d_list <= 10).sum() * 100 / len(topk_obj_2d_list)
        rel_acc_1 = (topk_rel_list <= 1).sum() * 100 / len(topk_rel_list)
        rel_acc_2d_1 = (topk_rel_2d_list <= 1).sum() * 100 / len(topk_rel_2d_list)
        rel_acc_3 = (topk_rel_list <= 3).sum() * 100 / len(topk_rel_list)
        rel_acc_2d_3 = (topk_rel_2d_list <= 3).sum() * 100 / len(topk_rel_2d_list)
        rel_acc_5 = (topk_rel_list <= 5).sum() * 100 / len(topk_rel_list)
        rel_acc_2d_5 = (topk_rel_2d_list <= 5).sum() * 100 / len(topk_rel_2d_list)
        # triplet_acc_50 = (topk_triplet_list <= 50).sum() * 100 / len(topk_triplet_list)
        # triplet_acc_2d_50 = (topk_triplet_2d_list <= 50).sum() * 100 / len(topk_triplet_2d_list)
        # triplet_acc_100 = (topk_triplet_list <= 100).sum() * 100 / len(topk_triplet_list)
        # triplet_acc_2d_100 = (topk_triplet_2d_list <= 100).sum() * 100 / len(topk_triplet_2d_list)
        recall_sgcls_gc = np.stack(recall_sgcls_gc).mean(0)
        recall_sgcls_ngc = np.stack(recall_sgcls_ngc).mean(0)
        recall_predcls_gc = np.stack(recall_predcls_gc).mean(0)
        recall_predcls_ngc = np.stack(recall_predcls_ngc).mean(0)

        # rel_acc_mean_1, rel_acc_mean_3, rel_acc_mean_5 = self.compute_mean_predicate(cls_matrix_list, topk_rel_list)
        # rel_acc_2d_mean_1, rel_acc_2d_mean_3, rel_acc_2d_mean_5 = self.compute_mean_predicate(cls_matrix_list, topk_rel_2d_list)

     
        print(f"Eval: 3d obj Acc@1  : {obj_acc_1}", file=f_in)   
        print(f"Eval: 2d obj Acc@1: {obj_acc_2d_1}", file=f_in)
        print(f"Eval: 3d obj Acc@5  : {obj_acc_5}", file=f_in) 
        print(f"Eval: 2d obj Acc@5: {obj_acc_2d_5}", file=f_in)  
        print(f"Eval: 3d obj Acc@10 : {obj_acc_10}", file=f_in)  
        print(f"Eval: 2d obj Acc@10: {obj_acc_2d_10}", file=f_in)
        print(f"Eval: 3d rel Acc@1  : {rel_acc_1}", file=f_in) 
        #print(f"Eval: 3d mean rel Acc@1  : {rel_acc_mean_1}", file=f_in)   
        print(f"Eval: 2d rel Acc@1: {rel_acc_2d_1}", file=f_in)
        #print(f"Eval: 2d mean rel Acc@1: {rel_acc_2d_mean_1}", file=f_in)
        print(f"Eval: 3d rel Acc@3  : {rel_acc_3}", file=f_in)   
        #print(f"Eval: 3d mean rel Acc@3  : {rel_acc_mean_3}", file=f_in) 
        print(f"Eval: 2d rel Acc@3: {rel_acc_2d_3}", file=f_in)
        #print(f"Eval: 2d mean rel Acc@3: {rel_acc_2d_mean_3}", file=f_in)
        print(f"Eval: 3d rel Acc@5  : {rel_acc_5}", file=f_in)
        #print(f"Eval: 3d mean rel Acc@5  : {rel_acc_mean_5}", file=f_in) 
        print(f"Eval: 2d rel Acc@5: {rel_acc_2d_5}", file=f_in)
        #print(f"Eval: 2d mean rel Acc@5: {rel_acc_2d_mean_5}", file=f_in)
        #print(f"Eval: 3d triplet Acc@50 : {triplet_acc_50}", file=f_in)
        #print(f"Eval: 2d triplet Acc@50: {triplet_acc_2d_50}", file=f_in)
        #print(f"Eval: 3d triplet Acc@100 : {triplet_acc_100}", file=f_in)
        #print(f"Eval: 2d triplet Acc@100: {triplet_acc_2d_100}", file=f_in)
        #print(f"Eval: 3d mean recall@50 : {mean_recall[0]}", file=f_in)
        #print(f"Eval: 2d mean recall@50: {mean_recall_2d[0]}", file=f_in)
        #print(f"Eval: 3d mean recall@100 : {mean_recall[1]}", file=f_in)
        #print(f"Eval: 2d mean recall@100: {mean_recall_2d[1]}", file=f_in)
        #print(f"Eval: 3d zero-shot recall@50 : {zero_shot_recall[0]}", file=f_in)
        #print(f"Eval: 3d zero-shot recall@100: {zero_shot_recall[1]}", file=f_in)
        #print(f"Eval: 3d non-zero-shot recall@50 : {non_zero_shot_recall[0]}", file=f_in)
        #print(f"Eval: 3d non-zero-shot recall@100: {non_zero_shot_recall[1]}", file=f_in)
        #print(f"Eval: 3d all-zero-shot recall@50 : {all_zero_shot_recall[0]}", file=f_in)
        #print(f"Eval: 3d all-zero-shot recall@100: {all_zero_shot_recall[1]}", file=f_in)
        print(f"Eval :3d sgcls gc recall@20 :{recall_sgcls_gc[0]}")
        print(f"Eval :3d sgcls gc recall@50 :{recall_sgcls_gc[1]}")
        print(f"Eval :3d sgcls gc recall@100 :{recall_sgcls_gc[2]}")
        print(f"Eval :3d sgcls ngc recall@20 :{recall_sgcls_ngc[0]}")
        print(f"Eval :3d sgcls ngc recall@50 :{recall_sgcls_ngc[1]}")
        print(f"Eval :3d sgcls ngc recall@100 :{recall_sgcls_ngc[2]}")
        print(f"Eval :3d predcls gc recall@20 :{recall_predcls_gc[0]}")
        print(f"Eval :3d predcls gc recall@50 :{recall_predcls_gc[1]}")
        print(f"Eval :3d predcls gc recall@100 :{recall_predcls_gc[2]}")
        print(f"Eval :3d predcls ngc recall@20 :{recall_predcls_ngc[0]}")
        print(f"Eval :3d predcls ngc recall@50 :{recall_predcls_ngc[1]}")
        print(f"Eval :3d predcls ngc recall@100 :{recall_predcls_ngc[2]}")



        if self.model.config.EVAL:
            f_in.close()
        
        logs = [("Acc@1/obj_cls_acc", obj_acc_1),
                ("Acc@1/obj_2d_cls_acc", obj_acc_2d_1),
                ("Acc@5/obj_cls_acc", obj_acc_5),
                ("Acc@5/obj_2d_cls_acc", obj_acc_2d_5),
                ("Acc@10/obj_cls_acc", obj_acc_10),
                ("Acc@10/obj_2d_cls_acc", obj_acc_2d_10),
                ("Acc@1/rel_cls_acc", rel_acc_1),
                #("Acc@1/rel_cls_acc_mean", rel_acc_mean_1),
                ("Acc@1/rel_2d_cls_acc", rel_acc_2d_1),
                #("Acc@1/rel_2d_cls_acc_mean", rel_acc_2d_mean_1),
                ("Acc@3/rel_cls_acc", rel_acc_3),
                #("Acc@3/rel_cls_acc_mean", rel_acc_mean_3),
                ("Acc@3/rel_2d_cls_acc", rel_acc_2d_3),
                #("Acc@3/rel_2d_cls_acc_mean", rel_acc_2d_mean_3),
                ("Acc@5/rel_cls_acc", rel_acc_5),
                #("Acc@5/rel_cls_acc_mean", rel_acc_mean_5),
                ("Acc@5/rel_2d_cls_acc", rel_acc_2d_5),
                # ("Acc@5/rel_2d_cls_acc_mean", rel_acc_2d_mean_5),
                # ("Acc@50/triplet_acc", triplet_acc_50),
                # ("Acc@50/triplet_2d_acc", triplet_acc_2d_50),
                # ("Acc@100/triplet_acc", triplet_acc_100),
                # ("Acc@100/triplet_2d_acc", triplet_acc_2d_100),
                # ("mean_recall@50", mean_recall[0]),
                # ("mean_2d_recall@50", mean_recall_2d[0]),
                # ("mean_recall@100", mean_recall[1]),
                # ("mean_2d_recall@100", mean_recall_2d[1]),
                # ("zero_shot_recall@50", zero_shot_recall[0]),
                # ("zero_shot_recall@100", zero_shot_recall[1]),
                # ("non_zero_shot_recall@50", non_zero_shot_recall[0]),
                # ("non_zero_shot_recall@100", non_zero_shot_recall[1]),
                # ("all_zero_shot_recall@50", all_zero_shot_recall[0]),
                # ("all_zero_shot_recall@100", all_zero_shot_recall[1])

                ]
        
        self.log(logs, self.model.iteration)
        #return mean_recall[0]
        return 0

    def validation3(self, debug_mode = False):
        val_loader = CustomDataLoader(
            config = self.config,
            dataset=self.dataset_valid,
            batch_size=1,
            num_workers=self.config.WORKERS,
            drop_last=False,
            shuffle=False,
            collate_fn=collate_fn_mmg
        )
       
        total = len(self.dataset_valid)
        progbar = op_utils.Progbar(total, width=20, stateful_metrics=['Misc/it'])
        
        print('===   start evaluation   ===')
        self.model.eval()
        topk_obj_list, topk_rel_list, topk_triplet_list, cls_matrix_list, edge_feature_list = np.array([]), np.array([]), np.array([]), [], []
        sub_scores_list, obj_scores_list, rel_scores_list = [], [], []
        topk_obj_2d_list, topk_rel_2d_list, topk_triplet_2d_list = np.array([]), np.array([]), np.array([])
        recall_predcls_gc, recall_predcls_ngc, recall_sgcls_gc, recall_sgcls_ngc = dict(), dict(), dict(), dict()
        # initial for recall dict
        for _ in range(26):
            recall_predcls_gc[_] = [[] for _ in range(3)]
            recall_predcls_ngc[_] = [[] for _ in range(3)]
            recall_sgcls_gc[_] = [[] for _ in range(3)]
            recall_sgcls_ngc[_] = [[] for _ in range(3)]
        
        for i, items in enumerate(val_loader, 0):
            ''' get data '''
            obj_points, obj_2d_feats, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids = self.data_processing_val(items)            
            
            with torch.no_grad():
                # if self.model.config.EVAL:
                #     top_k_obj, top_k_rel, tok_k_triplet, cls_matrix, sub_scores, obj_scores, rel_scores \
                #         = self.model.process_val(obj_points, gt_class, descriptor, gt_rel_cls, edge_indices, use_triplet=True)
                # else:
                #top_k_obj, top_k_obj_2d, top_k_rel, top_k_rel_2d, tok_k_triplet, top_k_2d_triplet, cls_matrix, sub_scores, obj_scores, rel_scores \
                top_k_obj, top_k_obj_2d, top_k_rel, top_k_rel_2d, predcls_gc, predcls_ngc, sgcls_gc, sgcls_ngc  \
                    = self.model.process_val3(obj_points, obj_2d_feats, gt_class, descriptor, gt_rel_cls, edge_indices, batch_ids, use_triplet=True)
                        
            ''' calculate metrics '''
            topk_obj_list = np.concatenate((topk_obj_list, top_k_obj))
            topk_obj_2d_list = np.concatenate((topk_obj_2d_list, top_k_obj_2d))
            topk_rel_list = np.concatenate((topk_rel_list, top_k_rel))
            topk_rel_2d_list = np.concatenate((topk_rel_2d_list, top_k_rel_2d))
            #topk_triplet_list = np.concatenate((topk_triplet_list, tok_k_triplet))
            #topk_triplet_2d_list = np.concatenate((topk_triplet_2d_list, top_k_2d_triplet))
            # if cls_matrix is not None:
            #     cls_matrix_list.extend(cls_matrix)
            #     sub_scores_list.extend(sub_scores)
            #     obj_scores_list.extend(obj_scores)
            #     rel_scores_list.extend(rel_scores)
            for i in range(26):
                for j in range(3):
                    if predcls_gc[i][j]!=-1:
                        recall_predcls_gc[i][j].append(predcls_gc[i][j]) 
                    if predcls_ngc[i][j]!=-1:
                        recall_predcls_ngc[i][j].append(predcls_ngc[i][j])
                    if sgcls_gc[i][j]!=-1:
                        recall_sgcls_gc[i][j].append(sgcls_gc[i][j])
                    if sgcls_ngc[i][j]!=-1:
                        recall_sgcls_ngc[i][j].append(sgcls_ngc[i][j])

            mR20_predcls_gc = np.mean([np.mean(recall_predcls_gc[i][0]) for i in range(26) if len(recall_predcls_gc[i][0])>0])
            mR20_predcls_ngc = np.mean([np.mean(recall_predcls_ngc[i][0]) for i in range(26) if len(recall_predcls_ngc[i][0])>0])
            mR20_sgcls_gc = np.mean([np.mean(recall_sgcls_gc[i][0]) for i in range(26) if len(recall_sgcls_gc[i][0])>0 ])
            mR20_sgcls_ngc = np.mean([np.mean(recall_sgcls_ngc[i][0]) for i in range(26) if len(recall_sgcls_ngc[i][0])>0])
            mR50_predcls_gc = np.mean([np.mean(recall_predcls_gc[i][1]) for i in range(26) if len(recall_predcls_gc[i][1])>0])
            mR50_predcls_ngc = np.mean([np.mean(recall_predcls_ngc[i][1]) for i in range(26) if len(recall_predcls_ngc[i][1])>0 ])
            mR50_sgcls_gc = np.mean([np.mean(recall_sgcls_gc[i][1]) for i in range(26) if len(recall_sgcls_gc[i][1])>0 ])
            mR50_sgcls_ngc = np.mean([np.mean(recall_sgcls_ngc[i][1]) for i in range(26) if len(recall_sgcls_ngc[i][1])>0 ])
            mR100_predcls_gc = np.mean([np.mean(recall_predcls_gc[i][2]) for i in range(26) if len(recall_predcls_gc[i][2])>0 ])
            mR100_predcls_ngc = np.mean([np.mean(recall_predcls_ngc[i][2]) for i in range(26) if len(recall_predcls_ngc[i][2])>0 ])
            mR100_sgcls_gc = np.mean([np.mean(recall_sgcls_gc[i][2]) for i in range(26) if len(recall_sgcls_gc[i][2])>0])
            mR100_sgcls_ngc = np.mean([np.mean(recall_sgcls_ngc[i][2]) for i in range(26) if len(recall_sgcls_ngc[i][2])>0])
            
            
            logs = [("Acc@1/obj_cls_acc", (topk_obj_list <= 1).sum() * 100 / len(topk_obj_list)),
                    ("Acc@1/obj_cls_2d_acc", (topk_obj_2d_list <= 1).sum() * 100 / len(topk_obj_2d_list)),
                    ("Acc@5/obj_cls_acc", (topk_obj_list <= 5).sum() * 100 / len(topk_obj_list)),
                    ("Acc@5/obj_cls_2d_acc", (topk_obj_2d_list <= 5).sum() * 100 / len(topk_obj_2d_list)),
                    ("Acc@10/obj_cls_acc", (topk_obj_list <= 10).sum() * 100 / len(topk_obj_list)),
                    ("Acc@10/obj_cls_2d_acc", (topk_obj_2d_list <= 10).sum() * 100 / len(topk_obj_2d_list)),
                    ("Acc@1/rel_cls_acc", (topk_rel_list <= 1).sum() * 100 / len(topk_rel_list)),
                    ("Acc@1/rel_cls_2d_acc", (topk_rel_2d_list <= 1).sum() * 100 / len(topk_rel_2d_list)),
                    ("Acc@3/rel_cls_acc", (topk_rel_list <= 3).sum() * 100 / len(topk_rel_list)),
                    ("Acc@3/rel_cls_2d_acc", (topk_rel_2d_list <= 3).sum() * 100 / len(topk_rel_2d_list)),
                    ("Acc@5/rel_cls_acc", (topk_rel_list <= 5).sum() * 100 / len(topk_rel_list)),
                    ("Acc@5/rel_cls_2d_acc", (topk_rel_2d_list <= 5).sum() * 100 / len(topk_rel_2d_list)),
                    ("mR20/predcls_gc", mR20_predcls_gc),
                    ("mR20/predcls_ngc", mR20_predcls_ngc),
                    ("mR20/sgcls_gc", mR20_sgcls_gc),
                    ("mR20/sgcls_ngc", mR20_sgcls_ngc),
                    ("mR50/predcls_gc", mR50_predcls_gc),
                    ("mR50/predcls_ngc", mR50_predcls_ngc),
                    ("mR50/sgcls_gc", mR50_sgcls_gc),
                    ("mR50/sgcls_ngc", mR50_sgcls_ngc),
                    ("mR100/predcls_gc", mR100_predcls_gc),
                    ("mR100/predcls_ngc", mR100_predcls_ngc),
                    ("mR100/sgcls_gc", mR100_sgcls_gc),
                    ("mR100/sgcls_ngc", mR100_sgcls_ngc)]
                    #("Acc@50/triplet_acc", (topk_triplet_list <= 50).sum() * 100 / len(topk_triplet_list)),
                    #("Acc@50/triplet_2d_acc", (topk_triplet_2d_list <= 50).sum() * 100 / len(topk_triplet_2d_list)),
                    #("Acc@100/triplet_acc", (topk_triplet_list <= 100).sum() * 100 / len(topk_triplet_list)),
                    #("Acc@100/triplet_2d_acc", (topk_triplet_2d_list <= 100).sum() * 100 / len(topk_triplet_2d_list)),]

            progbar.add(1, values=logs if self.config.VERBOSE else [x for x in logs if not x[0].startswith('Loss')])


        # cls_matrix_list = np.stack(cls_matrix_list)
        # sub_scores_list = np.stack(sub_scores_list)
        # obj_scores_list = np.stack(obj_scores_list)
        # rel_scores_list = np.stack(rel_scores_list)
        # mean_recall = get_mean_recall(topk_triplet_list, cls_matrix_list)
        # mean_recall_2d = get_mean_recall(topk_triplet_2d_list, cls_matrix_list)
        # zero_shot_recall, non_zero_shot_recall, all_zero_shot_recall = get_zero_shot_recall(topk_triplet_list, cls_matrix_list, self.dataset_valid.classNames, self.dataset_valid.relationNames)
        
        if self.model.config.EVAL:
            save_path = os.path.join(self.config.PATH, "results", self.model_name, self.exp)
            os.makedirs(save_path, exist_ok=True)
            # np.save(os.path.join(save_path,'topk_pred_list.npy'), topk_rel_list )
            # np.save(os.path.join(save_path,'topk_triplet_list.npy'), topk_triplet_list)
            # np.save(os.path.join(save_path,'cls_matrix_list.npy'), cls_matrix_list)
            # np.save(os.path.join(save_path,'sub_scores_list.npy'), sub_scores_list)
            # np.save(os.path.join(save_path,'obj_scores_list.npy'), obj_scores_list)
            # np.save(os.path.join(save_path,'rel_scores_list.npy'), rel_scores_list)
            f_in = open(os.path.join(save_path, 'result2.txt'), 'w')
        else:
            f_in = None   
        
        obj_acc_1 = (topk_obj_list <= 1).sum() * 100 / len(topk_obj_list)
        obj_acc_2d_1 = (topk_obj_2d_list <= 1).sum() * 100 / len(topk_obj_2d_list)
        obj_acc_5 = (topk_obj_list <= 5).sum() * 100 / len(topk_obj_list)
        obj_acc_2d_5 = (topk_obj_2d_list <= 5).sum() * 100 / len(topk_obj_2d_list)
        obj_acc_10 = (topk_obj_list <= 10).sum() * 100 / len(topk_obj_list)
        obj_acc_2d_10 = (topk_obj_2d_list <= 10).sum() * 100 / len(topk_obj_2d_list)
        rel_acc_1 = (topk_rel_list <= 1).sum() * 100 / len(topk_rel_list)
        rel_acc_2d_1 = (topk_rel_2d_list <= 1).sum() * 100 / len(topk_rel_2d_list)
        rel_acc_3 = (topk_rel_list <= 3).sum() * 100 / len(topk_rel_list)
        rel_acc_2d_3 = (topk_rel_2d_list <= 3).sum() * 100 / len(topk_rel_2d_list)
        rel_acc_5 = (topk_rel_list <= 5).sum() * 100 / len(topk_rel_list)
        rel_acc_2d_5 = (topk_rel_2d_list <= 5).sum() * 100 / len(topk_rel_2d_list)
        # triplet_acc_50 = (topk_triplet_list <= 50).sum() * 100 / len(topk_triplet_list)
        # triplet_acc_2d_50 = (topk_triplet_2d_list <= 50).sum() * 100 / len(topk_triplet_2d_list)
        # triplet_acc_100 = (topk_triplet_list <= 100).sum() * 100 / len(topk_triplet_list)
        # triplet_acc_2d_100 = (topk_triplet_2d_list <= 100).sum() * 100 / len(topk_triplet_2d_list)
        mR20_predcls_gc = np.mean([np.mean(recall_predcls_gc[i][0]) for i in range(26) if len(recall_predcls_gc[i][0])>0])
        mR20_predcls_ngc = np.mean([np.mean(recall_predcls_ngc[i][0]) for i in range(26) if len(recall_predcls_ngc[i][0])>0])
        mR20_sgcls_gc = np.mean([np.mean(recall_sgcls_gc[i][0]) for i in range(26) if len(recall_sgcls_gc[i][0])>0 ])
        mR20_sgcls_ngc = np.mean([np.mean(recall_sgcls_ngc[i][0]) for i in range(26) if len(recall_sgcls_ngc[i][0])>0])
        mR50_predcls_gc = np.mean([np.mean(recall_predcls_gc[i][1]) for i in range(26) if len(recall_predcls_gc[i][1])>0])
        mR50_predcls_ngc = np.mean([np.mean(recall_predcls_ngc[i][1]) for i in range(26) if len(recall_predcls_ngc[i][1])>0 ])
        mR50_sgcls_gc = np.mean([np.mean(recall_sgcls_gc[i][1]) for i in range(26) if len(recall_sgcls_gc[i][1])>0 ])
        mR50_sgcls_ngc = np.mean([np.mean(recall_sgcls_ngc[i][1]) for i in range(26) if len(recall_sgcls_ngc[i][1])>0 ])
        mR100_predcls_gc = np.mean([np.mean(recall_predcls_gc[i][2]) for i in range(26) if len(recall_predcls_gc[i][2])>0 ])
        mR100_predcls_ngc = np.mean([np.mean(recall_predcls_ngc[i][2]) for i in range(26) if len(recall_predcls_ngc[i][2])>0 ])
        mR100_sgcls_gc = np.mean([np.mean(recall_sgcls_gc[i][2]) for i in range(26) if len(recall_sgcls_gc[i][2])>0])
        mR100_sgcls_ngc = np.mean([np.mean(recall_sgcls_ngc[i][2]) for i in range(26) if len(recall_sgcls_ngc[i][2])>0])
        # rel_acc_mean_1, rel_acc_mean_3, rel_acc_mean_5 = self.compute_mean_predicate(cls_matrix_list, topk_rel_list)
        # rel_acc_2d_mean_1, rel_acc_2d_mean_3, rel_acc_2d_mean_5 = self.compute_mean_predicate(cls_matrix_list, topk_rel_2d_list)

     
        print(f"Eval: 3d obj Acc@1  : {obj_acc_1}", file=f_in)   
        print(f"Eval: 2d obj Acc@1: {obj_acc_2d_1}", file=f_in)
        print(f"Eval: 3d obj Acc@5  : {obj_acc_5}", file=f_in) 
        print(f"Eval: 2d obj Acc@5: {obj_acc_2d_5}", file=f_in)  
        print(f"Eval: 3d obj Acc@10 : {obj_acc_10}", file=f_in)  
        print(f"Eval: 2d obj Acc@10: {obj_acc_2d_10}", file=f_in)
        print(f"Eval: 3d rel Acc@1  : {rel_acc_1}", file=f_in) 
        #print(f"Eval: 3d mean rel Acc@1  : {rel_acc_mean_1}", file=f_in)   
        print(f"Eval: 2d rel Acc@1: {rel_acc_2d_1}", file=f_in)
        #print(f"Eval: 2d mean rel Acc@1: {rel_acc_2d_mean_1}", file=f_in)
        print(f"Eval: 3d rel Acc@3  : {rel_acc_3}", file=f_in)   
        #print(f"Eval: 3d mean rel Acc@3  : {rel_acc_mean_3}", file=f_in) 
        print(f"Eval: 2d rel Acc@3: {rel_acc_2d_3}", file=f_in)
        #print(f"Eval: 2d mean rel Acc@3: {rel_acc_2d_mean_3}", file=f_in)
        print(f"Eval: 3d rel Acc@5  : {rel_acc_5}", file=f_in)
        #print(f"Eval: 3d mean rel Acc@5  : {rel_acc_mean_5}", file=f_in) 
        print(f"Eval: 2d rel Acc@5: {rel_acc_2d_5}", file=f_in)
        #print(f"Eval: 2d mean rel Acc@5: {rel_acc_2d_mean_5}", file=f_in)
        #print(f"Eval: 3d triplet Acc@50 : {triplet_acc_50}", file=f_in)
        #print(f"Eval: 2d triplet Acc@50: {triplet_acc_2d_50}", file=f_in)
        #print(f"Eval: 3d triplet Acc@100 : {triplet_acc_100}", file=f_in)
        #print(f"Eval: 2d triplet Acc@100: {triplet_acc_2d_100}", file=f_in)
        #print(f"Eval: 3d mean recall@50 : {mean_recall[0]}", file=f_in)
        #print(f"Eval: 2d mean recall@50: {mean_recall_2d[0]}", file=f_in)
        #print(f"Eval: 3d mean recall@100 : {mean_recall[1]}", file=f_in)
        #print(f"Eval: 2d mean recall@100: {mean_recall_2d[1]}", file=f_in)
        #print(f"Eval: 3d zero-shot recall@50 : {zero_shot_recall[0]}", file=f_in)
        #print(f"Eval: 3d zero-shot recall@100: {zero_shot_recall[1]}", file=f_in)
        #print(f"Eval: 3d non-zero-shot recall@50 : {non_zero_shot_recall[0]}", file=f_in)
        #print(f"Eval: 3d non-zero-shot recall@100: {non_zero_shot_recall[1]}", file=f_in)
        #print(f"Eval: 3d all-zero-shot recall@50 : {all_zero_shot_recall[0]}", file=f_in)
        #print(f"Eval: 3d all-zero-shot recall@100: {all_zero_shot_recall[1]}", file=f_in)
        print(f"Eval :3d sgcls gc mrecall@20 : {mR20_sgcls_gc}", file=f_in)
        print(f"Eval :3d sgcls ngc mrecall@20 : {mR20_sgcls_ngc}", file=f_in)
        print(f"Eval :3d predcls gc mrecall@20 : {mR20_predcls_gc}", file=f_in)
        print(f"Eval :3d predcls ngc mrecall@20 : {mR20_predcls_ngc}", file=f_in)
        print(f"Eval :3d sgcls gc mrecall@50 : {mR50_sgcls_gc}", file=f_in)
        print(f"Eval :3d sgcls ngc mrecall@50 : {mR50_sgcls_ngc}", file=f_in)
        print(f"Eval :3d predcls gc mrecall@50 : {mR50_predcls_gc}", file=f_in)
        print(f"Eval :3d predcls ngc mrecall@50 : {mR50_predcls_ngc}", file=f_in)
        print(f"Eval :3d sgcls gc mrecall@100 : {mR100_sgcls_gc}", file=f_in)
        print(f"Eval :3d sgcls ngc mrecall@100 : {mR100_sgcls_ngc}", file=f_in)
        print(f"Eval :3d predcls gc mrecall@100 : {mR100_predcls_gc}", file=f_in)
        print(f"Eval :3d predcls ngc mrecall@100 : {mR100_predcls_ngc}", file=f_in)


        if self.model.config.EVAL:
            f_in.close()
        
        logs = [("Acc@1/obj_cls_acc", obj_acc_1),
                ("Acc@1/obj_2d_cls_acc", obj_acc_2d_1),
                ("Acc@5/obj_cls_acc", obj_acc_5),
                ("Acc@5/obj_2d_cls_acc", obj_acc_2d_5),
                ("Acc@10/obj_cls_acc", obj_acc_10),
                ("Acc@10/obj_2d_cls_acc", obj_acc_2d_10),
                ("Acc@1/rel_cls_acc", rel_acc_1),
                #("Acc@1/rel_cls_acc_mean", rel_acc_mean_1),
                ("Acc@1/rel_2d_cls_acc", rel_acc_2d_1),
                #("Acc@1/rel_2d_cls_acc_mean", rel_acc_2d_mean_1),
                ("Acc@3/rel_cls_acc", rel_acc_3),
                #("Acc@3/rel_cls_acc_mean", rel_acc_mean_3),
                ("Acc@3/rel_2d_cls_acc", rel_acc_2d_3),
                #("Acc@3/rel_2d_cls_acc_mean", rel_acc_2d_mean_3),
                ("Acc@5/rel_cls_acc", rel_acc_5),
                #("Acc@5/rel_cls_acc_mean", rel_acc_mean_5),
                ("Acc@5/rel_2d_cls_acc", rel_acc_2d_5),
                # ("Acc@5/rel_2d_cls_acc_mean", rel_acc_2d_mean_5),
                # ("Acc@50/triplet_acc", triplet_acc_50),
                # ("Acc@50/triplet_2d_acc", triplet_acc_2d_50),
                # ("Acc@100/triplet_acc", triplet_acc_100),
                # ("Acc@100/triplet_2d_acc", triplet_acc_2d_100),
                # ("mean_recall@50", mean_recall[0]),
                # ("mean_2d_recall@50", mean_recall_2d[0]),
                # ("mean_recall@100", mean_recall[1]),
                # ("mean_2d_recall@100", mean_recall_2d[1]),
                # ("zero_shot_recall@50", zero_shot_recall[0]),
                # ("zero_shot_recall@100", zero_shot_recall[1]),
                # ("non_zero_shot_recall@50", non_zero_shot_recall[0]),
                # ("non_zero_shot_recall@100", non_zero_shot_recall[1]),
                # ("all_zero_shot_recall@50", all_zero_shot_recall[0]),
                # ("all_zero_shot_recall@100", all_zero_shot_recall[1])

                ]
        
        self.log(logs, self.model.iteration)
        #return mean_recall[0]
        return 0

    
    def compute_mean_predicate(self, cls_matrix_list, topk_pred_list):
        cls_dict = {}
        for i in range(26):
            cls_dict[i] = []
        
        for idx, j in enumerate(cls_matrix_list):
            if j[-1] != -1:
                cls_dict[j[-1]].append(topk_pred_list[idx])
        
        predicate_mean_1, predicate_mean_3, predicate_mean_5 = [], [], []
        for i in range(26):
            l = len(cls_dict[i])
            if l > 0:
                m_1 = (np.array(cls_dict[i]) <= 1).sum() / len(cls_dict[i])
                m_3 = (np.array(cls_dict[i]) <= 3).sum() / len(cls_dict[i])
                m_5 = (np.array(cls_dict[i]) <= 5).sum() / len(cls_dict[i])
                predicate_mean_1.append(m_1)
                predicate_mean_3.append(m_3)
                predicate_mean_5.append(m_5) 
           
        predicate_mean_1 = np.mean(predicate_mean_1)
        predicate_mean_3 = np.mean(predicate_mean_3)
        predicate_mean_5 = np.mean(predicate_mean_5)

        return predicate_mean_1 * 100, predicate_mean_3 * 100, predicate_mean_5 * 100
