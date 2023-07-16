#from features import *
from curses import meta
from pickletools import string1
from re import A
from sys import path_importer_cache
import DataStruct as ds
import visualization as vis
#from sklearn.decomposition import IncrementalPCA
import moviepy.editor as mp
import numpy as np
from typing import List
import scipy
import matplotlib.pyplot as plt
import csv
import pandas as pd
import random
import json
import os
from tqdm import tqdm

conn=ds.Connectivity().load(skeleton_path='/hpc/group/tdunn/hk276/CAPTURE_demo/CAPTURE_data/skeletons.py', skeleton_name='NTUhuman')
path_to_skels='/hpc/group/tdunn/action_data/PKU-MMD-unzipped/skeletons/' 
# skipA = []
skipV=[]
# skipA = [i for i in range(1,121) if i>29]  
skipA = []

def PKU_plot_files(file_names: List = [], name: str="temp.mp4", rgb: bool=False, avgsmooth: bool=False):
    skels = 0
    if type(file_names[0])==str:
        file_names=[[i] for i in file_names]
    for i in range(len(file_names)):
        data=[]
        for k in range(len(file_names[i])):
            try:
                data.append(np.load(path_to_skels+file_names[i][k]+'.skeleton.npy',allow_pickle=True).item()["skel_body0"][:,:,[0,2,1]])
                data[k]=data[k]-np.average(data[k][0,:,:], axis=0)
            except:
                print("file "+file_names[i][k]+" not found")
        if data==[]:
            continue
        data=align_skels(data)
        length=len(data[0])
        if avgsmooth:
            nvideos=1
            avg = np.average(data,axis=0)
            dis=5
            for j in range(len(avg[i])-1):
                avg[j]=np.average(avg[j:min(j+dis,len(avg)-1)], axis=0)
            data=avg
        else:
            nvideos=len(data)
            for j in range(1,len(data)):
                data[0]=np.append(data[0],data[j], axis=0)
            data=data[0]
        vis.skeleton_vid3D(data,
                        connectivity = conn,
                        title=act_class[int(file_names[i][0][-3:])-1],
                        frames=[int(length/2)+length*i for i in range(nvideos)],
                        N_FRAMES = length-2,
                        fps=30,
                        VID_NAME=file_names[i][0]+"HumanSkel.mp4",
                        SAVE_ROOT = "./videos/",
                        fsize=25,
                        aspect=[0.9,0.9,1.5],
                        offset=[0.1,0.1])
        if skels==0:
            skels=mp.VideoFileClip("./videos/vis_"+file_names[i][0]+"HumanSkel.mp4")
        else:
            skels=mp.concatenate_videoclips([skels,mp.VideoFileClip("./videos/vis_"+file_names[i][0]+"HumanSkel.mp4")])
    if rgb:
        clip = mp.VideoFileClip(path_to_vids+file_names[i][0][1:4]+"/"+file_names[0][0]+'_rgb.avi')
        for i in range(1,len(file_names)):
            nextclip=mp.VideoFileClip(path_to_vids+file_names[i][0][1:4]+"/"+file_names[i][0]+'_rgb.avi')
            clip=mp.concatenate_videoclips([clip,nextclip])
        #clip=mp.clips_array([skels,clip])
        clip=mp.CompositeVideoClip([skels.set_position(("left","center")).resize(0.30), clip.set_position(("right","center"))],size=(3000,1080))
        clip.write_videofile('./videos/'+name, codec='libx264')
    else:
        skels.write_videofile('./videos/'+name, codec='libx264')

def get_raw_PKU():
    files=os.listdir("/hpc/group/tdunn/action_data/PKU-MMD-unzipped/skeletons/")
    meta=[]
    labels=[]
    data=[]
    dic={"L":0,"M":1,"R":2}
    currframe=0
    for fi in tqdm(files):
        meta.append([int(fi[1:3]),int(fi[4:6]),dic[fi[7]]]) # action group, subject, view
        f=open('/hpc/group/tdunn/action_data/PKU-MMD-unzipped/skeletons/'+fi,'r')
        read=csv.reader(f,delimiter=' ')
        data.append([])
        data[-1].append(np.reshape(np.float_(next(read))[:75],(25,3)))
        for i in read:
            data[-1].append(np.reshape(np.float_(i[1:])[:75],(25,3)))
        f.close()
        data[-1]=np.array(data[-1])[:,:,[0,2,1]]
        f=open('/hpc/group/tdunn/action_data/PKU-MMD-unzipped/labels/'+fi,'r')
        read=csv.reader(f,delimiter=',')
        row=np.int_(np.array([i for i in read]))[:,:3]
        row[:,1:3]+=currframe
        if len(labels)!=0:
            labels=np.concatenate((labels,row))
        else:
            labels=row
        currframe+=len(data[-1])
    
    np.save("/hpc/group/tdunn/action_data/PKUmeta.npy", meta)
    np.save("/hpc/group/tdunn/action_data/PKUdata.npy", data)
    np.save("/hpc/group/tdunn/action_data/PKUlabels.npy", labels)

    return [data,meta,labels]

def make_dict(PKUfull,meta):
    datadict={}
    for j in range(len(meta)):
        datadict["".join(["%03d"%(i) for i in meta[j]])] = PKUfull[j]
    return datadict


def PKU_to_mat(PKUfull,meta):
    # smoothPKU=np.load("/hpc/group/tdunn/action_data/SmoothedData.npy", allow_pickle=True)
    smoothPKU=PKUfull
    # print(np.shape(smoothPKU))
    exp_id=np.concatenate([np.full(len(smoothPKU[i]),i) for i in range(len(smoothPKU))])
    # PKUdata=np.load("/hpc/group/tdunn/action_data/FullData.npy", allow_pickle=True)
    # meta=np.load("/hpc/group/tdunn/action_data/FullDataMeta.npy", allow_pickle=True)

    # datadict=makedict(PKUfull,meta)

    with open('PKUmeta.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["action","person","view"])
        writer.writerows(meta)

    smoothPKU=np.swapaxes(np.concatenate(smoothPKU),0,1) # now its joint, frame, coord
    matout={'video':exp_id}
    joints=['Base',
            'SpineM',
            'Neck',
            'Head',
            'ShoulderL',
            'ElbowL',
            'WristL',
            'HandL',
            'ShoulderR',
            'ElbowR',
            'WristR',
            'HandR',
            'HipL',
            'KneeL',
            'AnkleL',
            'FootL',
            'HipR',
            'KneeR',
            'AnkleR',
            'FootR',
            'SpineF',
            'FingerL',
            'ThumbL',
            'FingerR',
            'ThumbR']
    preddict={}
    for i in range(len(smoothPKU)):
        preddict[joints[i]]=smoothPKU[i]
    matout['predictions']=preddict
    print(np.shape(smoothPKU))
    scipy.io.savemat("PKU.mat",matout)

[data,meta,labels]=get_raw_PKU()
PKU_to_mat(data,meta)
# import pdb;pdb.set_trace();
# vis.skeleton_vid3D(data,
#                         connectivity = conn,
#                         title="ble",
#                         frames=[300],
#                         N_FRAMES = 200,
#                         fps=30,
#                         VID_NAME="PKUtest.mp4",
#                         SAVE_ROOT = "./videos/",
#                         fsize=25,
#                         aspect=[0.9,0.9,1.5],
#                         offset=[0.1,0.1])