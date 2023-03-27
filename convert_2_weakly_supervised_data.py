import json
from tqdm import tqdm
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

import clip
clip_model, preprocess = clip.load('ViT-B/32', device='cuda:0')

def generate_descritpion(object_dict):
    description = 'This is a {}. '.format(object_dict['label'])
    if "attributes" in object_dict:
        for attribute in object_dict['attributes']:
            att_str  = ', '.join(object_dict['attributes'][attribute])
            description += 'Its {} is {}. '.format(attribute, att_str)
    
    if "affordances" in object_dict:
        aff_preposition_end = []
        aff_no_preposition = []
        for aff in object_dict['affordances']:
            if aff.split(' ')[-1] in ['on', 'in', 'with', 'under', 'against']:
                aff_preposition_end.append(aff + ' it')
            else:
                aff_no_preposition.append(aff)

        aff_preposition_end_str = ', '.join(aff_preposition_end)
        aff_no_preposition_str = ', '.join(aff_no_preposition)
        
        if len(aff_preposition_end_str) > 0:
            description += 'We can {}. '.format(aff_preposition_end_str)
        if len(aff_no_preposition_str) > 0:
            description += 'It can be used for {}. '.format(aff_no_preposition_str)
    
    if "state_affordances" in object_dict:
        aff_str = ','.join(object_dict['state_affordances'])
        description += 'It can be {}. '.format(aff_str)
    
    return description.strip()

def get_2d_features(scene_id, objects_ids, object_json):
    image_path = '/data/caidaigang/project/3DSSG_Repo/data/3RScan/{}/multi_view/instance_{}_class_{}_origin_view_mean.npy'
    images_features = []
    for i in objects_ids:
        images_features.append(np.load(image_path.format(scene_id, i, object_json[i])).reshape(-1))
    return torch.tensor(np.array(images_features)).cuda()

def compile_json(rel_json_path, object_json_path):
    rel_json = json.load(open(rel_json_path))['scans']
    object_json = json.load(open(object_json_path))['scans']
    scans_2_objects = {}
    print('Compiling object json...')
    for i in tqdm(object_json):
        assert i['scan'] not in scans_2_objects
        scans_2_objects[i['scan']] = {}
        for j in i['objects']:
            scans_2_objects[i['scan']][j['id']] = generate_descritpion(j)
    
    # compile object json
    new_rel_json = {
        'scans':[]
    }

    print('Compiling relationship json...')

    correct_num, all_num = 0, 0
    for i in tqdm(rel_json):
        if i["scan"] == 'fa79392f-7766-2d5c-869a-f5d6cfb62fc6':
            continue
        
        objects = i['objects']
        objects_ids = np.array(list(objects.keys()))
        objects_2d_features = get_2d_features(i['scan'], objects_ids, objects)
        obj_prompt = torch.cat([clip.tokenize(scans_2_objects[i['scan']][j]) for j in objects_ids]).cuda()

        with torch.no_grad():
            rel_text_features = clip_model.encode_text(obj_prompt)
        
        # import ipdb; ipdb.set_trace()
        compute_similarity = rel_text_features @ objects_2d_features.T
        row, col = linear_sum_assignment(compute_similarity.cpu(), maximize=True)
        predict_obj_id = objects_ids[col]
        
        correct_num += np.sum(predict_obj_id == objects_ids)
        all_num += len(objects_ids)

        tmp = {
            'scan': i['scan'],
            'objects': objects,
            'split':i['split'],
            'relationships':[]
        }
        for j in i['relationships']:
            if str(j[0]) in objects and str(j[1]) in objects:
                 tmp['relationships'].append([predict_obj_id[objects_ids == str(j[0])].astype(np.int32).item(), \
                                              predict_obj_id[objects_ids == str(j[1])].astype(np.int32).item(), j[2], j[3]])
        
        new_rel_json['scans'].append(tmp)
    
    print('Accuracy: {}'.format(correct_num / all_num))
    json.dump(new_rel_json, open(rel_json_path.replace('.json', '_weakly_new.json'), 'w'))


if __name__ == '__main__':
    rel_json_path = '/data/caidaigang/project/WS3DSSG/data/3DSSG_subset/relationships_train.json'
    object_json_path = '/data/caidaigang/project/WS3DSSG/data/3DSSG/3DSSG/objects.json'
    compile_json(rel_json_path, object_json_path)