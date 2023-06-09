# -*- coding: utf-8 -*-
#@title
import numpy as np
import torch
import torch.nn.functional as F
from utils.metrics import box_inter, box_iou
import copy
import time

def enclosing_canvas(img, boxes, where_old_image=None, where_new_image=None, prev_out=None, background = 0.3, minimum= 33):
    '''     
    <Description>
    This method is forming a enclosing canvas.  Enclosing canvas is a box within the image that encloses all patch locations.
    
    <input> 
    img : image
    boxes : bounding boxes
    where_old_image : patch locations in the image
    where_new_image : patch locations in the canvas
    prev_out : previous bounding boxes
    background : d% background
    
    <output>
    packed frame. 
    patch locations in the image, 
    None, 
    groups of each contains bounding bax in the patches
    '''
    
    
    groups = [[x for x in range(len(boxes))]]
    if where_old_image==None:
        pass
    elif len(where_old_image)==1:
        x, y, z, w = where_old_image[0,1], where_old_image[0,3], where_old_image[0,0], where_old_image[0,2]
        return img[:,:,x:y, z:w], where_old_image, None, groups
    min_len, max_len = max(img.size(3), img.size(2))//minimum, max(img.size(3), img.size(2))//16
    boxes = expand_box(boxes, expand = background, thres_len= (min_len, max_len), x_size = img.size(3), y_size = img.size(2), prev_out=prev_out)
    boxes= torch.cat([torch.min(boxes[...,:2], dim=0)[0], torch.max(boxes[...,2:], dim=0)[0]],dim=0)

    boxes[0].clamp_(min=0, max=img.size(3))
    boxes[1].clamp_(min=0, max=img.size(2))
    boxes[2].clamp_(min=0, max=img.size(3))
    boxes[3].clamp_(min=0, max=img.size(2))
    boxes = boxes.type(torch.cuda.LongTensor)
    
    new_input = img[:,:,boxes[1]:boxes[3],boxes[0]:boxes[2]]

    return new_input, boxes.unsqueeze(0), None , groups

def patch_construction(img, boxes, prev_out=None, background=0.3, minimum=33):
    '''     
    <Description>
    This method is a patch construction.
    
    <input> 
    img : image
    boxes : bounding boxes
    prev_out : previous bounding boxes
    background : d% background
    
    <output>
    new_box : patches
    groups : groups of each contains bounding bax in the patches
    '''    
    
    
    # Sort by width ratio
    for_sort_labels = boxes.clone()
    min_len, max_len = max(img.size(3), img.size(2))//minimum, max(img.size(3), img.size(2))//16
    for_sort_labels = expand_box(for_sort_labels, expand = background, thres_len= (min_len, max_len), x_size = img.size(3), y_size = img.size(2), prev_out=prev_out)
    for_sort_labels[:,[0,2]].clamp_(min=0, max=img.size(3))
    for_sort_labels[:,[1,3]].clamp_(min=0, max=img.size(2))
    inter_matrix = ((box_iou(for_sort_labels, for_sort_labels)>0.12)*1).cpu().numpy()
    for_sort_labels = for_sort_labels.cpu().numpy()
    groups = [(i+np.nonzero(inter_matrix[i,i:])[0]).tolist() for i in range(len(inter_matrix))]
    
    finish =0
    while finish != len(groups)-1:
        group = groups[finish]
        to_add =[]
        for idx, check_group in enumerate(groups[finish+1:]):
            for num in group:
                if num in check_group:
                    to_add.append(finish+1+idx)
        to_add = list(set(to_add))
        if len(to_add) ==0:
            finish+=1
        else:
            for i in sorted(to_add, reverse=True):
                groups[finish].extend(copy.deepcopy(groups[i]))
                del groups[i]
            groups[finish] = list(set(groups[finish]))


    box_to_crop = [np.concatenate([np.min(for_sort_labels[group,0:2],axis=0, keepdims=True), np.max(for_sort_labels[group,2:4],axis=0, keepdims=True)], axis=1) for group in groups]
    new_box = np.concatenate(box_to_crop, axis=0)
    while True:
        temp = torch.from_numpy(new_box)
        inter_matrix = ((box_iou(temp, temp)>0.12)*1).numpy()
        if np.sum(inter_matrix) == len(inter_matrix):
            break
        else:
            groups = [(i+np.nonzero(inter_matrix[i,i:])[0]).tolist() for i in range(len(inter_matrix))]
            finish =0
            while finish != len(groups)-1:
                group = groups[finish]
                to_add =[]
                for idx, check_group in enumerate(groups[finish+1:]):
                    for num in group:
                        if num in check_group:
                            to_add.append(finish+1+idx)
                to_add = list(set(to_add))
                if len(to_add) ==0:
                    finish+=1
                else:
                    for i in sorted(to_add, reverse=True):
                        groups[finish].extend(copy.deepcopy(groups[i]))
                        del groups[i]
                    groups[finish] = list(set(groups[finish]))
            box_to_crop = [np.concatenate([np.min(new_box[group,0:2],axis=0, keepdims=True), np.max(new_box[group,2:4],axis=0, keepdims=True)], axis=1) for group in groups]
            new_box = np.concatenate(box_to_crop, axis=0)


    # 묶을 그룹별로 정리
    new_box[:, [1,3]] = np.clip(new_box[:, [1,3]], 0, img.size(2))
    new_box[:, [0,2]] = np.clip(new_box[:, [0,2]], 0, img.size(3))
    return new_box.astype(np.int32), groups

import cv2
#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
def canvas(img, boxes, where_old_image=None,where_new_image=None, prev_out=None, background = 0.3, minimum=33):
    '''     
    <Description>
    This method is forming a DynaPP canvas. 
    
    <input> 
    img : image
    boxes : bounding boxes
    where_old_image : patch locations in the image
    where_new_image : patch locations in the canvas
    prev_out : previous bounding boxes
    background : d% background
    
    <output>
    packed frame. 
    patch locations in the image, 
    patch locations in the canvas, 
    groups of each contains bounding bax in the patches
    '''    
    
    if where_old_image==None:
        if len(boxes)==1:
            pack2_img, pack2_where_old_image, pack2_where_new_image, pack_groups_groups = enclosing_canvas(img, boxes, background=background, minimum=minimum)
            return pack2_img, pack2_where_old_image, pack2_where_new_image, pack_groups_groups
        else:
            where_old_image, groups_groups = patch_construction(img, boxes, prev_out=prev_out, background=background, minimum=minimum)
    elif len(where_old_image)==1:
        x, y, z, w = where_old_image[0,1], where_old_image[0,3], where_old_image[0,0], where_old_image[0,2]
        return img[:,:,x:y, z:w], where_old_image, None, None
    else:
        w_max, h_max = torch.max(where_new_image[:,2:4], dim=0)[0]
        img_comb = torch.full((img.size(0), img.size(1), h_max, w_max), 0.).to(img.device).half()  # base image with 4 tiles
        for i in range(len(where_old_image)):
            img_comb[:,:,where_new_image[i,1]:where_new_image[i,3], where_new_image[i,0]:where_new_image[i,2]] = img[:,:, where_old_image[i,1]:where_old_image[i,3],where_old_image[i,0]:where_old_image[i,2]]
        return img_comb, where_old_image, where_new_image, None


    images =[]
    for i in range(len(where_old_image)):
        x, y, z, w = where_old_image[i,1], where_old_image[i,3], where_old_image[i,0], where_old_image[i,2]
        images.append(img[:,:,x:y, z:w])

    if len(where_old_image)==1:
        return images[0], torch.from_numpy(where_old_image).to(boxes.device), None, groups_groups

    shapes = np.vstack([np.array([image.size(2),image.size(3)]) for image in images]) #  height, width
    # 라벨이미지 생성


    index = np.vstack([[x] for x in range(len(where_old_image))])
    where_new_image= where_old_image-np.concatenate([where_old_image[:,:2],where_old_image[:,:2]],axis=1)
    where_new_image = np.concatenate([index,where_new_image], axis = 1)
        
    
    shapes[:,0][shapes[:,0]==0] = 1
    shapes[:,1][shapes[:,1]==0] = 1   
    area = shapes[:,0]*shapes[:,1]
    area_order = area.argsort()

    sorted_area = area[area_order]
    cuts = []
    cut_thres = sorted_area[0]
    for i in range(len(sorted_area)-1):
        if 1.3*cut_thres < sorted_area[i+1]:
            cuts.append(i+1)
            cut_thres = sorted_area[i+1]
    cuts.append(len(sorted_area))
    # 너무 큰 이미지는 분리


    previous_cut = 0
    size_sort_shapes = []
    size_sort_where_new_image = []
    for cut in cuts:
        size_sort_shapes.append(shapes[area_order[previous_cut:cut],:])
        size_sort_where_new_image.append([where_new_image[idx:idx+1,:] for idx in area_order[previous_cut:cut]])
        previous_cut =  cut
    
    finish = 0
    finish_time = len(cuts)
    shapes = size_sort_shapes[finish]
    where_new_images = size_sort_where_new_image[finish]

    threshold = []        
    previous_cut =  0
    for i, cut in enumerate(cuts):
        if i !=0:
            threshold.append(np.mean(area[area_order[previous_cut:cut]]))
        previous_cut =  cut
    threshold.append(1e10) # infinite

    done = True
    while finish != finish_time:
        all_where_new_images = []
        if finish !=0:
            shapes = np.concatenate([size_sort_shapes[finish],shapes], 0)
            where_new_images = size_sort_where_new_image[finish]+where_new_images
        while len(shapes) > 1:
            shapes[:,0][shapes[:,0]==0] = 1
            shapes[:,1][shapes[:,1]==0] = 1
            height_large = shapes[:,0]/shapes[:,1] > 1
            width_large = shapes[:,1]/shapes[:,0] >= 1
            if len(shapes) ==2 and width_large[0]!=width_large[1]:
                if max(shapes[0,0], shapes[1,0])*(shapes[0,1]+shapes[1,1]) > max(shapes[0,1], shapes[1,1])*(shapes[0,0]+shapes[1,0]):
                    width_large = [True, True]
                    height_large = [False, False]
                else:
                    width_large = [False, False]
                    height_large = [True, True]                   
            wid_shape, hei_shape = shapes[width_large,:], shapes[height_large,:]
            wid_where_new_image, hei_where_new_image = [where_new_images[idx] for idx, accord in enumerate(width_large) if accord==True], [where_new_images[idx] for idx, accord in enumerate(width_large) if accord==False]
            shapes = np.empty([0,2])
            where_new_images = []
            for id_, (sort_shape, sort_where_new_images) in enumerate([(wid_shape,wid_where_new_image), (hei_shape,hei_where_new_image)]):
                if len(sort_shape) ==0:
                    continue

                irect = sort_shape[:,id_].argsort()
                sort_where_new_images = [sort_where_new_images[idx] for idx in irect]
                sort_shape = sort_shape[irect,:]

                all_coord = np.empty([0,4])
                i = 0
                for idxes, shape in enumerate(sort_shape):                        
                    done = False
                    # Load image                        
                    h, w = shape

                    # place img in img9
                    if id_ ==0:
                        if idxes %2== 0:  # center
                            h0, w0 = h, w
                            c = np.array([[0, 0, w, h]])  # xmin, ymin, xmax, ymax (base) coordinates
                            i = 1
                            where_new_images.append(sort_where_new_images[idxes])
                        else:  # top
                            c  = np.array([[0, hp, w, hp+h]])
                            i = 0
                            sort_where_new_images[idxes] += np.array([[0, 0, hp, 0, hp]], dtype=np.int32)
                            where_new_images[-1] = np.concatenate([where_new_images[-1],sort_where_new_images[idxes]], axis=0)
                            done = True
                    else:
                        if idxes %2 == 0:  # center
                            h0, w0 = h, w
                            c = np.array([[0, 0, w, h]])  # xmin, ymin, xmax, ymax (base) coordinates
                            i = 1
                            where_new_images.append(sort_where_new_images[idxes])
                        else:  # top
                            c  = np.array([[wp, 0, wp+w, h]])
                            i = 0
                            sort_where_new_images[idxes] += np.array([[0, wp, 0, wp, 0]], dtype=np.int32)
                            where_new_images[-1] = np.concatenate([where_new_images[-1],sort_where_new_images[idxes]], axis=0)
                            done = True
                  

                    hp, wp = h, w  # height, width previous

                    all_coord = np.concatenate([all_coord,c], axis=0)
                    # print(all_coord)
                    if done == True:
                        # all_coord = all_coord.astype(np.uint16)
                        w_max, h_max = np.max(all_coord[:,2:], axis=0).astype(np.uint16)
                        shapes = np.concatenate([shapes, np.array([[h_max,w_max]])], 0)
                        all_coord = np.empty([0,4])
                    
                if done == False:
                    # all_coord = all_coord.astype(np.uint16)
                    w_max, h_max = int(all_coord[0,2]),int(all_coord[0,3])
                    shapes = np.concatenate([shapes, np.array([[h_max,w_max]])], 0)


            shapes[:,0][shapes[:,0]==0] = 1
            shapes[:,1][shapes[:,1]==0] = 1
            if np.max(shapes[:,0]*shapes[:,1]) >= threshold[finish]:
                break
        finish +=1


    where_new_image = where_new_images[0]
    i_image = where_new_image[:,0].argsort()
    where_new_image = where_new_image[i_image,:]
    where_new_image = torch.from_numpy(where_new_image[:,1:]).to(boxes.device)
    where_old_image = torch.from_numpy(where_old_image).to(boxes.device)

    w_max, h_max = torch.max(where_new_image[:,2:4], dim=0)[0]
    img9 = torch.full((img.size(0), img.size(1), h_max, w_max), 0.).to(img.device).half()  # base image with 4 tiles
    for i in range(len(where_old_image)):
        img9[:,:,where_new_image[i,1]:where_new_image[i,3], where_new_image[i,0]:where_new_image[i,2]] = img[:,:, where_old_image[i,1]:where_old_image[i,3],where_old_image[i,0]:where_old_image[i,2]]
    

    # assert True ==False, f'{where_old_image}\n\n\n\n{where_new_image}\n\n\n\n\n\n{where_old_image-np.concatenate([where_old_image[:,0:2],where_old_image[:,0:2]], axis=1)}\n\n\n\n{where_new_image[:,1:]-np.concatenate([where_new_image[:,1:3],where_new_image[:,1:3]], axis=1)}'
    pack2_img, pack2_where_old_image, pack2_where_new_image, pack_groups_groups = enclosing_canvas(img, boxes, background = background, minimum= minimum)
    if pack2_img.size(2)*pack2_img.size(2) <= img9.size(2)*img9.size(3):
        return pack2_img, pack2_where_old_image, pack2_where_new_image, pack_groups_groups
    else:
        return img9, where_old_image, where_new_image, groups_groups


def expand_box(boxes, expand, thres_len, x_size, y_size, prev_out):
    '''     
    <Description>
    This method appends background to ROIs/bounding boxes:
    
    <input> 
    boxes : bounding boxes
    expand : d% background
    thres_len : [m%, o%]
    x_size, y_size : image width, height
    prev_out : previous bounding boxes
    
    <output>
    boxes: expanded ROIs
    '''    
    edge_thres = max(x_size,y_size)//100
    length_x = (boxes[:,2:3] - boxes[:,0:1])*expand/2
    length_y =  (boxes[:,3:4] - boxes[:,1:2])*expand/2
    

    # length_x[torch.where((boxes[:,0:1] < 10) + (boxes[:,2:3]> x_size-10) + (boxes[:,1:2]<10) + (boxes[:,3:4] > y_size-10) >0)] = thres_len[1]
    # length_y[length_x==thres_len[1]] = thres_len[1]
    length_x = torch.clamp(length_x, min=thres_len[0])
    length_y = torch.clamp(length_y, min=thres_len[0])
    boxes[:,0:1],boxes[:,1:2],boxes[:,2:3],boxes[:,3:4] = boxes[:,0:1]-length_x,boxes[:,1:2]-length_y,boxes[:,2:3]+length_x,boxes[:,3:4]+length_y
    return boxes
