import os
import json
import torch
import torch.optim as optim
import numpy as np
import random
from torch.utils.data import DataLoader
from tqdm import tqdm
import datetime
from utils.init import init
from dataset import ScannetReferenceDataset
from utils.init import CONF
from models.less import Unet, Compute_loss, Compute_iou
from transformers import RobertaTokenizerFast
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

class Logger:
    def __init__(self, dirname, filename):
        self.dirname = os.path.join(dirname, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        if not os.path.exists(self.dirname):
            os.makedirs(self.dirname)
        self.logout = open(os.path.join(self.dirname, filename), 'a')
        
    def log_string(self, out_str):
        if dist.get_rank() == 0:
            self.logout.write(out_str + '\n')
            self.logout.flush()
            print(out_str)


def get_dataloader(args, scanrefer, all_scene_list, split, config):
    dataset = ScannetReferenceDataset(
        scanrefer=scanrefer, 
        scanrefer_all_scene=all_scene_list, 
        split=split, 
        sample_points=args.sample_points, 
        augment=(not args.no_augment),
        cfg=config
    )

    train_sampler = DistributedSampler(dataset)
    if split == 'val':
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, \
                                collate_fn=dataset.collate_fn, drop_last=True, sampler=train_sampler) 
    else:
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, \
                                collate_fn=dataset.collate_fn, drop_last=True, sampler=train_sampler) 
    return dataset, dataloader


def get_num_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = int(sum([np.prod(p.size()) for p in model_parameters]))

    return num_params


def get_scanrefer(num_scenes):

    scanrefer_train = json.load(open(os.path.join(CONF.PATH.DATA, "scanrefer/ScanRefer_filtered_train.json")))
    scanrefer_val = json.load(open(os.path.join(CONF.PATH.DATA, "scanrefer/ScanRefer_filtered_val.json")))

    # get initial scene list
    train_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_train])))
    val_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_val])))
    if num_scenes == -1: 
        num_scenes = len(train_scene_list)
    else:
        assert len(train_scene_list) >= num_scenes
    
    # slice train_scene_list
    train_scene_list = train_scene_list[:num_scenes]

    # filter data in chosen scenes
    new_scanrefer_train = []
    for data in scanrefer_train:
        if data["scene_id"] in train_scene_list:
            new_scanrefer_train.append(data)

    new_scanrefer_val = scanrefer_val

    # all scanrefer scene
    all_scene_list = train_scene_list + val_scene_list

    print("train on {} samples and val on {} samples".format(len(new_scanrefer_train), len(new_scanrefer_val)))

    return new_scanrefer_train, new_scanrefer_val, all_scene_list

def adjust_learning_rate(optimizer, epoch, cfg):
    lr = optimizer.param_groups[0]['lr']       
    lr = lr * cfg.lr_decays[epoch]                  
    optimizer.param_groups[0]['lr'] = lr
    
    lr = optimizer.param_groups[1]['lr']
    lr = lr * cfg.lr_decays[epoch] 
    optimizer.param_groups[1]['lr'] = lr

def train_one_epoch(net, train_dataloader, optimizer, epoch, cfg, tokenizer, logger):
    
    adjust_learning_rate(optimizer, epoch, cfg)
    net.train()  # set model to training mode
    loss_cum = 0
    iou_cum = 0
    loss_mask_cum = 0
    loss_area_cum = 0
    loss_p2p_cum = 0
    length = len(train_dataloader)
    for step, data_dict in enumerate(train_dataloader):
        for key in data_dict:
            if key not in ['scene_id', 'lang_len', 'lang_token', 'description', 'spatial_shape', \
                           'lang_feat', 'batch_offset', 'unique_multiple']:
                if type(data_dict[key]) is list:
                    for i in range(len(data_dict[key])):
                        data_dict[key][i] = data_dict[key][i].cuda()
                else:
                    data_dict[key] = data_dict[key].cuda()
        
        lang_tokens = tokenizer(data_dict['lang_feat'], return_tensors='pt', padding=True)
        for name in lang_tokens.data:
            lang_tokens.data[name] = lang_tokens.data[name].cuda()
        data_dict['lang_tokens'] = lang_tokens
                
        optimizer.zero_grad()
        end_points = net(data_dict)
        
        loss, end_points = Compute_loss(end_points)
        loss.backward()
        optimizer.step()   
        

        iou, end_points, _, _ = Compute_iou(end_points)
        
        loss_cum += loss.detach().cpu().numpy()
        loss_mask_cum += end_points['loss_mask'].detach().cpu().numpy()
        loss_area_cum += end_points['loss_area'].detach().cpu().numpy()
        loss_p2p_cum += end_points['loss_p2p'].detach().cpu().numpy()

        iou_cum += iou
        log_interval = cfg.log_interval
        
        
        if (step+1) % log_interval == 0 and dist.get_rank() == 0:
            times = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            logger.log_string("[train][%d / %d] time:%s loss:%f loss_mask:%f loss_area:%f loss_p2p:%f miou:%f" % \
                      (step+1, length, times, (loss_cum / log_interval), (loss_mask_cum / log_interval), \
                      (loss_area_cum / log_interval), (loss_p2p_cum / log_interval), (iou_cum / log_interval)))
            loss_cum = 0
            loss_mask_cum = 0
            loss_area_cum = 0
            loss_p2p_cum = 0
            iou_cum = 0


def evaluate_one_epoch(net, val_dataloader, tokenizer, logger):
    
    net.eval()
    iou_cum = 0
    inter_counts = 0
    union_counts = 0
    iou_list = []
    unique_iou_list = []
    multiple_iou_list = []
    info_list = []
    for data_dict in tqdm(val_dataloader):
        for key in data_dict:
            if key not in ['scene_id', 'lang_len', 'lang_token', 'description', 'spatial_shape', \
                           'lang_feat', 'object_id', 'ann_id', 'unique_multiple', 'object_name', 'batch_offset']:
                if type(data_dict[key]) is list:
                    for i in range(len(data_dict[key])):
                        data_dict[key][i] = data_dict[key][i].cuda()
                else:
                    data_dict[key] = data_dict[key].cuda()
        
        
        lang_tokens = tokenizer(data_dict['lang_feat'], return_tensors='pt', padding=True)
        for name in lang_tokens.data:
            lang_tokens.data[name] = lang_tokens.data[name].cuda()
        data_dict['lang_tokens'] = lang_tokens
        
        
        with torch.no_grad():
            end_points = net(data_dict)

        
        iou, end_points, inter_count, union_count = Compute_iou(end_points)

        iou_cum += iou
        inter_counts += inter_count[0]
        union_counts += union_count[0]
        iou_list.append(iou)
        if data_dict['unique_multiple'][0]:
            multiple_iou_list.append(iou)
        else:
            unique_iou_list.append(iou)

        if args.save_pred:
            s = "sceneid:%s, objid:%s, annid:%s, step:%d, iou:%f, type:%d, name:%s" % \
                (data_dict['scene_id'][0], data_dict['object_id'][0], data_dict['ann_id'][0], \
                 step, iou, data_dict['unique_multiple'][0], data_dict['object_name'][0])
            info_list.append(s)
            name = "val_sample_%4d" % step
            pred_mask = torch.sigmoid(data_dict['logits'][0]).detach().cpu().numpy() > 0.5
            outputs_npy_pred = pred_mask.astype(np.int64)
            np.save(args.dirname + '/cloud/' + name, outputs_npy_pred)
            
        step += 1

    if args.save_pred:
        np.savetxt('prediction.txt', info_list)

    # 算一下acc25和acc50
    Precision_5= (np.array(iou_list) > 0.5).sum().astype(float)/len(iou_list)
    Precision_25 = (np.array(iou_list) > 0.25).sum().astype(float)/len(iou_list)

    if dist.get_rank() == 0:
        logger.log_string("*********************************************")
        logger.log_string("val_log:")
        logger.log_string("whole iter mean iou (overall):%f" % (iou_cum / len(val_dataloader)))
        logger.log_string("whole iter overall iou (overall):%f" % (inter_counts / union_counts))
        logger.log_string("whole iter mean iou (unique):%f" % (sum(unique_iou_list) / len(unique_iou_list)))
        logger.log_string("whole iter mean iou (multiple):%f" % (sum(multiple_iou_list) / len(multiple_iou_list)))
        logger.log_string("whole iter Acc@25:%f" % Precision_25)
        logger.log_string("whole iter Acc@50:%f" % Precision_5)
        logger.log_string("*********************************************")
    
    return (iou_cum / len(val_dataloader))


def main(args, logger):
    # init training dataset
    print("preparing data...")
    scanrefer_train, scanrefer_val, all_scene_list = get_scanrefer(args.num_scenes)

    # dataloader
    _, train_dataloader = get_dataloader(args, scanrefer_train, all_scene_list, "train", args)
    _, val_dataloader = get_dataloader(args, scanrefer_val, all_scene_list, "val", args)

    print("initializing...")
    net = Unet(cfg=args)
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            # synBN
            net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net).cuda()
        else:
            net = net.cuda()
    
    other_param = list()
    bert_param = list()
    for name, param in net.named_parameters():
        if 'language_encoder' in name:
            bert_param.append(param)
        else:
            other_param.append(param)
        
    # model parameters
    params = get_num_params(net) / 1e6 
    if dist.get_rank() == 0: logger.log_string('#Params: %.1fM' % (params))
    
    optimizer = optim.AdamW([{'params': other_param, 'lr': args.learning_rate, 'weight_decay': args.wd},
                        {'params': bert_param, 'lr': args.bert_learning_rate}])
    epoch = 1
    if args.use_checkpoint != '':
        logger.log_string("loading checkpoint from {}...".format(args.use_checkpoint))
        checkpoint = torch.load(os.path.join(args.use_checkpoint), map_location=torch.device('cpu'))    # in order to avoid out of memory
        net.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        
    net = DistributedDataParallel(net, device_ids=[args.local_rank], broadcast_buffers=False, find_unused_parameters=True)
    
    print("Start training...\n")
    max_miou = 0
    now_miou = 0
    while(epoch <= args.max_epoch):
        logger.log_string('**** EPOCH %03d ****' % (epoch))
        train_one_epoch(net, train_dataloader, optimizer, epoch, args, tokenizer, logger)
        logger.log_string('**** EVAL EPOCH %03d START****' % (epoch))

        if (epoch) < 40:
            if (epoch) % 5 == 0:
                now_miou = evaluate_one_epoch(net, val_dataloader, tokenizer, logger)
            else:
                logger.log_string('skip eval')
        else:
            now_miou = evaluate_one_epoch(net, val_dataloader, tokenizer, logger)


        # Save checkpoint
        if dist.get_rank() == 0:
            save_dict = {'epoch': epoch+1, # after training one epoch, the start_epoch should be epoch+1
                        'optimizer_state_dict': optimizer.state_dict()}
            
            try: # with nn.DataParallel() the net is added as a submodule of DataParallel
                save_dict['model_state_dict'] = net.module.state_dict()
            except:
                save_dict['model_state_dict'] = net.state_dict()
            torch.save(save_dict, os.path.join(logger.dirname, 'checkpoint_last.pth'))
                
            if(now_miou>max_miou):       # 保存最好的iou的模型    
                torch.save(save_dict, os.path.join(logger.dirname, 'checkpoint_best.pth'))
                max_miou = now_miou

            logger.log_string('Best mIoU = {:2.2f}%'.format(max_miou*100))
            logger.log_string('**** EVAL EPOCH %03d END****' % (epoch))
            logger.log_string('')
        epoch = epoch + 1



if __name__ == "__main__":

    args = init()
    logger = Logger(args.output_dir, 'log.txt')
    
    # setting
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', timeout=datetime.timedelta(seconds=5400))  
    if args.tag is not None:
        logger.log_string("tag: %s" % args.tag)
    else:
        logger.log_string("no tag")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # to avoid tokenizer deadlock

    # reproducibility
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    main(args, logger)
    
