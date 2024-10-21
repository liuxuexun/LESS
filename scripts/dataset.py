import os
import sys
import time
import numpy as np
from torch.utils.data import Dataset
import torch
import pointgroup_ops

sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
from utils.init import CONF
from utils.pc_utils import random_sampling, rotx, roty, rotz
from data.scannet.model_util_scannet import ScannetDatasetConfig

# data setting
DC = ScannetDatasetConfig()
MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])

# data path
SCANNET_V2_TSV = os.path.join(CONF.PATH.SCANNET_META, "scannetv2-labels.combined.tsv")

class ScannetReferenceDataset(Dataset):
       
    def __init__(self, scanrefer, scanrefer_all_scene, 
        split="train", 
        sample_points=50000, 
        augment=False,
        cfg=None):

        self.scanrefer = scanrefer
        self.scanrefer_all_scene = scanrefer_all_scene # all scene_ids in scanrefer
        self.split = split
        self.sample_points = sample_points
        self.augment = augment
        self.cfg = cfg
        self.mode = 4
    
        # load data
        self._load_data()
       
    def __len__(self):
        return len(self.scanrefer)

    def __getitem__(self, idx):
        start = time.time()
        scene_id = self.scanrefer[idx]["scene_id"]
        object_id = int(self.scanrefer[idx]["object_id"])
        object_name = " ".join(self.scanrefer[idx]["object_name"].split("_"))
        ann_id = self.scanrefer[idx]["ann_id"]
        
        # get language features
        lang_feat = self.lang[scene_id][str(object_id)][ann_id]

        # get pc
        vertices = self.scene_data[scene_id]["vertices"].copy()
        instance_labels = self.scene_data[scene_id]["instance_labels"].copy()
        semantic_labels = self.scene_data[scene_id]["semantic_labels"].copy()
        

        point_cloud = vertices[:,0:6] 
        point_cloud[:,3:6] = (point_cloud[:,3:6]-MEAN_COLOR_RGB)/256.0
        feature = point_cloud[:,3:]
        
        if self.split != "val":
            point_cloud, choices = random_sampling(point_cloud, self.sample_points, return_choices=True)        
            instance_labels = instance_labels[choices]
            semantic_labels = semantic_labels[choices]
            feature = feature[choices]
        
        # ------------------------------- LABELS ------------------------------
        if self.split != "test" and self.split != "val":

            # ------------------------------- DATA AUGMENTATION ------------------------------        
            if self.augment:
                if np.random.random() > 0.5:
                    # Flipping along the YZ plane
                    point_cloud[:,0] = -1 * point_cloud[:,0]               
                    
                if np.random.random() > 0.5:
                    # Flipping along the XZ plane
                    point_cloud[:,1] = -1 * point_cloud[:,1]                             

                # Rotation along X-axis
                rot_angle = (np.random.random()*np.pi/18) - np.pi/36 # -5 ~ +5 degree
                rot_mat = rotx(rot_angle)
                point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat))

                # Rotation along Y-axis
                rot_angle = (np.random.random()*np.pi/18) - np.pi/36 # -5 ~ +5 degree
                rot_mat = roty(rot_angle)
                point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat))

                # Rotation along up-axis/Z-axis
                rot_angle = (np.random.random()*np.pi/18) - np.pi/36 # -5 ~ +5 degree
                rot_mat = rotz(rot_angle)
                point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat))

                # Translation
                point_cloud = self._translate(point_cloud)

        # sparseconv
        self.voxel_scale = 50
        self.voxel_spatial_shape = [128, 512]
        xyz_middle = point_cloud[:, :3]
        xyz = xyz_middle * self.voxel_scale
        xyz -= xyz.min(0)
        
        # binary label
        instance_refer_mask = instance_labels == object_id + 1

        data_dict = {}
        data_dict['scene_id'] = scene_id
        # data_dict["point_clouds"] = point_cloud.astype(np.float32) # point cloud data including features
        data_dict['coord'] = torch.from_numpy(xyz).long()
        data_dict['coord_float'] = torch.from_numpy(xyz_middle).float()
        
        
        data_dict["lang_feat"] = lang_feat
        data_dict['description'] = self.scanrefer[idx]['description']
        data_dict["scan_idx"] = np.array(idx).astype(np.int64)
        data_dict["feature"] = torch.from_numpy(feature).float()
        data_dict["object_id"] = np.array(int(object_id)).astype(np.int64)
        data_dict["ann_id"] = np.array(int(ann_id)).astype(np.int64)
        data_dict['instance_refer_mask'] = np.array(instance_refer_mask).astype(np.int64)
        data_dict["unique_multiple"] = np.array(self.unique_multiple_lookup[scene_id][str(object_id)][ann_id]).astype(np.int64)
        data_dict["load_time"] = time.time() - start
        
        data_dict['semantic_labels'] = semantic_labels.astype(np.int64)
        
        return data_dict
    
    def _get_raw2label(self):
        # mapping
        scannet_labels = DC.type2class.keys()
        scannet2label = {label: i for i, label in enumerate(scannet_labels)}

        lines = [line.rstrip() for line in open(SCANNET_V2_TSV)]
        lines = lines[1:]
        raw2label = {}
        for i in range(len(lines)):
            label_classes_set = set(scannet_labels)
            elements = lines[i].split('\t')
            raw_name = elements[1]
            nyu40_name = elements[7]
            if nyu40_name not in label_classes_set:
                raw2label[raw_name] = scannet2label['others']
            else:
                raw2label[raw_name] = scannet2label[nyu40_name]

        return raw2label

    def _get_unique_multiple_lookup(self):
        all_sem_labels = {}
        cache = {}
        for data in self.scanrefer:
            scene_id = data["scene_id"]
            object_id = data["object_id"]
            object_name = " ".join(data["object_name"].split("_"))
            ann_id = data["ann_id"]

            if scene_id not in all_sem_labels:
                all_sem_labels[scene_id] = []

            if scene_id not in cache:
                cache[scene_id] = {}

            if object_id not in cache[scene_id]:
                cache[scene_id][object_id] = {}
                try:
                    all_sem_labels[scene_id].append(self.raw2label[object_name])
                except KeyError:
                    all_sem_labels[scene_id].append(17)

        # convert to numpy array
        all_sem_labels = {scene_id: np.array(all_sem_labels[scene_id]) for scene_id in all_sem_labels.keys()}

        unique_multiple_lookup = {}
        for data in self.scanrefer:
            scene_id = data["scene_id"]
            object_id = data["object_id"]
            object_name = " ".join(data["object_name"].split("_"))
            ann_id = data["ann_id"]

            try:
                sem_label = self.raw2label[object_name]
            except KeyError:
                sem_label = 17

            unique_multiple = 0 if (all_sem_labels[scene_id] == sem_label).sum() == 1 else 1

            # store
            if scene_id not in unique_multiple_lookup:
                unique_multiple_lookup[scene_id] = {}

            if object_id not in unique_multiple_lookup[scene_id]:
                unique_multiple_lookup[scene_id][object_id] = {}

            if ann_id not in unique_multiple_lookup[scene_id][object_id]:
                unique_multiple_lookup[scene_id][object_id][ann_id] = None

            unique_multiple_lookup[scene_id][object_id][ann_id] = unique_multiple

        return unique_multiple_lookup

    def _load_text(self):
        lang = {}
        for data in self.scanrefer:
            scene_id = data["scene_id"]
            object_id = data["object_id"]
            ann_id = data["ann_id"]

            if scene_id not in lang:
                lang[scene_id] = {}

            if object_id not in lang[scene_id]:
                lang[scene_id][object_id] = {}

            if ann_id not in lang[scene_id][object_id]:
                lang[scene_id][object_id][ann_id] = {}

            # tokenize the description
            tokens = data["token"]
            embeddings = " ".join(tokens)    # add space
                
            # store
            lang[scene_id][object_id][ann_id] = embeddings

        return lang

    def _load_data(self):
        print("loading data...")
        # load language features
        self.lang = self._load_text()          

        # add scannet data
        self.scene_list = sorted(list(set([data["scene_id"] for data in self.scanrefer])))

        # load scene data
        self.scene_data = {}
        for scene_id in self.scene_list:
            self.scene_data[scene_id] = {}
            self.scene_data[scene_id]["vertices"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_vert.npy")
            self.scene_data[scene_id]["instance_labels"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_ins_label.npy")
            self.scene_data[scene_id]["semantic_labels"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_sem_label.npy")
            
        # prepare class mapping
        lines = [line.rstrip() for line in open(SCANNET_V2_TSV)]
        lines = lines[1:]
        raw2nyuid = {}
        for i in range(len(lines)):
            elements = lines[i].split('\t')
            raw_name = elements[1]
            nyu40_name = int(elements[4])
            raw2nyuid[raw_name] = nyu40_name


        self.raw2label = self._get_raw2label()
        self.unique_multiple_lookup = self._get_unique_multiple_lookup()

    def _translate(self, point_set):
        # unpack
        coords = point_set[:, :3]

        # translation factors
        x_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        y_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        z_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        factor = [x_factor, y_factor, z_factor]
        
        # dump
        coords += factor
        point_set[:, :3] = coords

        return point_set

    
    def collate_fn(self, batch):
        scene_ids, coords, coords_float, feats, instance_refer_masks, lang_feats, descriptions = [], [], [], [], [], [], []
        unique_multiples = []

        for i, data in enumerate(batch):

            scene_ids.append(data['scene_id'])
            coords.append(torch.cat([torch.LongTensor(data['coord'].shape[0], 1).fill_(i), data['coord']], 1))
            coords_float.append(data['coord_float'])
            feats.append(data['feature'])
            instance_refer_masks.append(torch.from_numpy(data['instance_refer_mask']).int().unsqueeze(0))
            lang_feats.append(data['lang_feat'])
            descriptions.append(data['description'])
            unique_multiples.append(data["unique_multiple"])



        coords = torch.cat(coords, 0)  # long [B*N, 1 + 3], the batch item idx is put in b_xyz[:, 0]
        coords_float = torch.cat(coords_float, 0)  # float [B*N, 3]
        feats = torch.cat(feats, 0)  # float [B*N, 3]
        instance_refer_masks = torch.cat(instance_refer_masks, 0)
            
        
        feats = torch.cat((coords_float, feats), dim=1)

        # voxelize
        spatial_shape = np.clip((coords.max(0)[0][1:] + 1).numpy(), self.voxel_spatial_shape[0], None)  # long [3]
        voxel_coords, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(coords, len(batch), self.mode)

        index = voxel_coords[:, 0]
        batch_offsets = [0]
        upper = 0
        for i in range(len(torch.unique(index))):
            upper += 50000
            batch_offsets.append(upper)

        if self.split == 'val':
            batch_offsets = [0, len(index)]

        return {
            'voxel_coords': voxel_coords,
            'p2v_map': p2v_map,
            'v2p_map': v2p_map,
            'spatial_shape': spatial_shape,
            'feats': feats,
            'scene_id': scene_ids,
            'instance_refer_mask': instance_refer_masks,
            'lang_feat': lang_feats,
            'description': descriptions,
            'batch_offset': batch_offsets,
            "unique_multiple": unique_multiples, 
        }