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

conn=ds.Connectivity().load(skeleton_path='/hpc/group/tdunn/hk276/CAPTURE_demo/CAPTURE_data/skeletons.py', skeleton_name='NTUhuman')
#act_class=["drink water", "eat meal/snack", "brushing teeth", "brushing hair", "drop", "pickup", "throw", "sitting down", "standing up", "clapping", "reading", "writing", "tear up paper", "wear jacket", "take off jacket", "wear a shoe", "take off a shoe", "wear on glasses", "take off glasses", "put on a hat/cap", "take off a hat/cap", "cheer up", "hand waving", "kicking something", "put something inside pocket", "hopping (one foot jumping)", "jump up", "make a phone call/answer phone", "playing with phone/tablet", "typing on a keyboard", "pointing to something with finger", "taking a selfie", "check time (from watch)", "rub two hands together", "nod head/bow", "shake head", "wipe face", "salute", "put the palms together", "cross hands in front (say stop)", "sneeze/cough", "staggering", "falling", "touch head (headache)", "touch chest (stomachache/heart pain)", "touch back (backache)", "touch neck (neckache)", "nausea or vomiting condition", "use a fan (with hand or paper)/feeling warm", "punching/slapping other person", "kicking other person", "pushing other person", "pat on back of other person", "point finger at the other person", "hugging other person", "giving something to other person", "touch other persons pocket", "handshaking", "walking towards each other", "walking apart from each other"]
act_class=["drink water","eat meal/snack","brushing teeth","brushing hair","drop","pickup","throw","sitting down","standing up (from sitting position)","clapping","reading","writing","tear up paper","wear jacket","take off jacket","wear a shoe","take off a shoe","wear on glasses","take off glasses","put on a hat/cap","take off a hat/cap","cheer up","hand waving","kicking something","reach into pocket","hopping (one foot jumping)","jump up","make a phone call/answer phone","playing with phone/tablet","typing on a keyboard","pointing to something with finger","taking a selfie","check time (from watch)","rub two hands together","nod head/bow","shake head","wipe face","salute","put the palms together","cross hands in front (say stop)","sneeze/cough","staggering","falling","touch head (headache)","touch chest (stomachache/heart pain)","touch back (backache)","touch neck (neckache)","nausea or vomiting condition","use a fan (with hand or paper)/feeling warm","punching/slapping other person","kicking other person","pushing other person","pat on back of other person","point finger at the other person","hugging other person","giving something to other person","touch other person\'s pocket","handshaking","walking towards each other","walking apart from each other","put on headphone","take off headphone","shoot at the basket","bounce ball","tennis bat swing","juggling table tennis balls","hush (quite)","flick hair","thumb up","thumb down","make ok sign","make victory sign","staple book","counting money","cutting nails","cutting paper (using scissors)","snapping fingers","open bottle","sniff (smell)","squat down","toss a coin","fold paper","ball up paper","play magic cube","apply cream on face","apply cream on hand back","put on bag","take off bag","put something into a bag","take something out of a bag","open a box","move heavy objects","shake fist","throw up cap/hat","hands up (both hands)","cross arms","arm circles","arm swings","running on the spot","butt kicks (kick backward)","cross toe touch","side kick","yawn","stretch oneself","blow nose","hit other person with something","wield knife towards other person","knock over other person (hit with body)","grab other person’s stuff","shoot at other person with a gun","step on foot","high-five","cheers and drink","carry something with other person","take a photo of other person","follow other person","whisper in other person’s ear","exchange things with other person","support somebody with hand","finger-guessing game (playing rock-paper-scissors)"]
path_to_skels='/hpc/group/tdunn/action_data/NTUskeletonsNP/' 
path_to_vids='/hpc/group/tdunn/action_data/nturgb+d_videos/nturgbd_rgb_s'
# skipA = []
skipV=[18119,18222,18446,18562,23367,26246]
skipV=[   73,   344,   671,   716,  1238,  1381,  1692,  2350,  4613,
        5332,  5877,  6036,  6195,  6706,  6761,  6790,  6794,  6837,
        6838,  6950,  7107,  7135,  7188,  7259,  7360,  7408,  7414,
        7442,  7494,  7518,  7552,  8197,  8413,  8633,  9423, 10269,
       10479, 12088, 12132, 12449, 12451, 12455, 12489, 12507, 12536,
       12606, 12684, 12698, 12803, 13073, 13078, 14589, 15153, 16383,
       18074, 18159, 22002, 22029, 32815, 32897, 32975, 34292, 34669,
       36342, 38742, 38858, 38864, 38910, 39046, 39082, 39088, 40034,
       40040, 40368, 40495, 42936, 46287, 46290, 46316, 46404, 46416,
       46422, 46428, 46482, 46554, 46622, 46674, 46734, 46736, 46746,
       46758, 46766, 46778, 46782, 46818, 46830, 46836, 46850, 46920,
       46953, 46965, 47031, 47072, 47085, 47102, 47154, 47250, 47364,
       47376, 47382, 47388, 47514, 47582, 47634, 47696, 47718, 47726,
       47742, 47778, 47796, 47880, 47913, 47925, 47991, 48032, 48039,
       48062, 48248, 48260, 48302, 48336, 48474, 48656, 48673, 48732,
       48795, 48835, 48840, 48873, 48885, 48951, 49159, 49208, 49220,
       49256, 49262, 49296, 49320, 49575, 49616, 49680, 49755, 49773,
       49833, 49845, 49911, 50142, 50204, 50255, 50703, 50709, 50756,
       50793, 50796, 50871, 51140, 51215, 51612, 51708, 51710, 51711,
       51714, 51722, 51756, 51831, 52040, 52572, 52608, 52682, 52791,
       52838, 52848, 53532, 53751, 53798, 53808, 54492, 54711, 54758,
       54760, 54768, 55452, 55671, 55718, 55728, 56412, 56631, 56678,
       56688, 57372, 57591, 57638, 57648, 58338, 58350, 58384, 58387,
       58392, 58404, 58476, 58500, 58608, 58769, 58868, 59027, 59028,
       59292, 59330, 59348, 59352, 59364, 59460, 59544, 59725, 59828,
       59906, 59930, 60090, 60252, 60307, 60308, 60310, 60311, 60312,
       60324, 60357, 60420, 60504, 60524, 60528, 60685, 60788, 61101,
       61256, 61272, 61279, 61284, 61317, 61380, 61488, 61538, 61748,
       62172, 62229, 62232, 62244, 62246, 62340, 62448, 62700, 62708,
       62736, 63204, 63225, 63234, 63300, 63660, 63668, 64164, 64260,
       65097, 65124, 65151, 65166, 65220, 65558, 66012, 66068, 66084,
       66092, 66117, 66156, 66972, 67027, 67028, 67030, 67044, 67082,
       67298, 67314, 67316, 67932, 68004, 68420, 68892, 68930, 68947,
       68948, 68949, 68950, 68964, 69036, 69168, 69900, 69924, 69945,
       70128, 70860, 70884, 71088, 71466, 71562, 71649, 71676, 71827,
       71830, 71869, 71882, 71928, 72048, 73731, 73739, 73775, 73785,
       75590, 75662, 75670, 75704, 76007, 76008, 76044, 76055, 76124,
       76160, 76205, 76257, 76381, 76384, 76405, 76406, 76412, 76414,
       76416, 76440, 76514, 76615, 76617, 76618, 76622, 76624, 76632,
       76633, 76659, 76672, 76686, 76699, 76701, 76708, 76753, 76754,
       76789, 76791, 76814, 76821, 76885, 76889, 76892, 76916, 77084,
       77577, 77592, 77633, 77960, 78060, 78531, 78537, 78552, 78750,
       78819, 78920, 79497, 79532, 79535, 79566, 79880, 80492, 80840,
       81452, 81800, 82760, 83720, 84002, 84335, 84680, 84962, 85119,
       85257, 85281, 85316, 85317, 85578, 85640, 85922, 86882, 88137,
       89097, 89418]
skipV=[]
# skipV=[1,2000,2222,7030,11800] # drink water, throw, sit, wave, salute
# skipA = [i for i in range(1,121) if i>29]  
skipA = [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120]
# skipA=skipA+[i for i in range(1,121) if i>60]

# pose_struct = ds.DataStruct(config_path = 'r../configs/path_configs/embedding_analysis_dcc_r01.yaml')
# pose_struct.load_connectivity("/hpc/group/tdunn/hk276/CAPTURE_demo/CAPTURE_data/skeletons.py", "NTUhuman")
# pose_struct.load_pose()
# pose_struct.load_feats_NTU()

# Separate videos have rotated floor planes - this rotates them back
# pose = align_floor(pose_struct.pose_3d, 
#                        pose_struct.exp_ids_full)

def NTU_plot_files(file_names: List = [], name: str="temp.mp4", rgb: bool=False, avgsmooth: bool=False):
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

'''
def NTU_plot_files(file_names: List = [], name: str="temp.mp4", rgb: bool=False):
    skels = 0
    if type(file_names[0])==str:
        file_names=[[i] for i in file_names]
    for i in range(len(file_names)):
        data=[]
        for k in range(len(file_names[i])):
            try:
                data.append(np.load(path_to_skels+file_names[i][k]+'.skeleton.npy',allow_pickle=True).item()["skel_body0"][:,:,[0,2,1]])
            except:
                print("file "+file_names[i][k]+" not found")
        if data==[]:
            continue
        data=align_skels(data)
        length=len(data[0])
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
'''

def align_skels(data: List=[]):
    n=5
    minlen=min([len(data[i]) for i in range(len(data))])
    # maxlen=max([len(data[i]) for i in range(len(data))])
    # global ndiff
    # global ntot
    # ntot+=1
    # if maxlen!=minlen:
    #     print(maxlen-minlen)
    #     ndiff+=1
    data=[i[0:minlen]-np.average(np.average(i[:,:,:], axis=1),axis=0) for i in data]
    for i in range(1,len(data)):
        rots=[]
        for j in range(-n,n):
            rots.append(scipy.spatial.transform.Rotation.align_vectors(data[0][j],data[i][j])[0])
        rots=[i.as_matrix() for i in rots]
        rots=np.average(rots, axis=0)
        rots=scipy.spatial.transform.Rotation.from_matrix(rots)
        for j in range(minlen):
            data[i][j]=rots.apply(data[i][j])
    return data
    
def err_by_act():
    AvgRmsdA=np.zeros(120)
    for A in range(1,121): #loop across A
        rmsds=[]
        print("############### starting A = "+str(A))
        if A in skipA:
            continue
        for S in range(1,33): #loop across S
            print("starting S = "+str(S))
            for P in range(1,107): #loop across P
                for R in range(1,3): #loop across R
                    data=[]
                    rmsdarr=np.zeros(0)
                    for C in range(3): # loop across cameras
                        try:
                            data.append(np.load(path_to_skels+'S%03dC%03dP%03dR%03dA%03d.skeleton.npy'%(S,C+1,P,R,A),allow_pickle=True).item()["skel_body0"][:,:,[0,2,1]])
                        except:
                            data=[]
                            break                    
                        data[C]=data[C]-np.average(data[C][0,:,:], axis=0) # video, frame, joint, coordinate
                        k=C-1
                        while k>=0:
                            for j in range(min(len(data[C]),len(data[k]))):
                                rmsdarr=np.append(rmsdarr,scipy.spatial.transform.Rotation.align_vectors(data[k][j],data[C][j])[1])
                            k=k-1
                    if(len(rmsdarr)>0):
                        rmsds.append(np.average(rmsdarr**2))
        AvgRmsdA[A-1]=np.average(rmsds)
        print(rmsds)
    print(AvgRmsdA)

def get_raw_NTU():
    vid=0
    NTUfull=[]
    meta=[]
    for A in range(1,121): #loop across action
        print("############### starting A = "+str(A))
        if A in skipA:
            continue
        for S in range(1,33): #loop across scenes
            print("starting S = "+str(S))
            for P in range(1,107): #loop across people
                for R in range(1,3): #loop across takes
                    data=[]
                    for C in range(3): # loop across cameras
                        try:
                            data=np.load(path_to_skels+'S%03dC%03dP%03dR%03dA%03d.skeleton.npy'%(S,C+1,P,R,A),allow_pickle=True).item()["skel_body0"][:,:,[0,2,1]]
                        except:
                            data=[]
                            continue
                        if vid in skipV:
                            vid+=1
                            continue
                        NTUfull.append(data)
                        meta.append([S,P,R,C,A])
                        vid+=1
    NTUfull=np.array(NTUfull)
    meta=np.array(meta)
    np.save("/hpc/group/tdunn/action_data/NTU60MetaViewsFull.npy", meta)
    np.save("/hpc/group/tdunn/action_data/NTU60ViewsFull.npy", NTUfull)
    return [NTUfull,meta]

def get_aligned_NTU(save=False, rand=999999):
    vid=0
    NTUfull=[]
    meta=[]
    for A in range(1,121): #loop across action
        print("############### starting A = "+str(A))
        if A in skipA:
            continue
        for S in range(1,33): #loop across scenes
            print("starting S = "+str(S))
            for P in range(1,107): #loop across people
                for R in range(1,3): #loop across takes
                    data=[]
                    if random.sample(range(0,37874),1)[0]>rand:
                        continue
                    for C in range(3): # loop across cameras
                        try:
                            data.append(np.load(path_to_skels+'S%03dC%03dP%03dR%03dA%03d.skeleton.npy'%(S,C+1,P,R,A),allow_pickle=True).item()["skel_body0"][:,:,[0,2,1]])
                        except:
                            data=[]
                            break   
                        if len(data)>0:
                            if vid not in skipV:
                                break 
                        else:
                            break                
                        data[C]=data[C]-np.average(data[C][0,:,:], axis=0) # video, frame, joint, coordinate
                        
                        # if C>0:
                        #     for j in range(min(len(data[C]),len(data[0]))):
                        #         rot=scipy.spatial.transform.Rotation.align_vectors(data[0][j],data[C][j])
                        #         data[C][j]=rot[0].apply(data[C][j])
                    if len(data)>0:
                        # minlen=min([len(data[i]) for i in range(len(data))])
                        # data=[i[0:minlen] for i in data]
                        if vid in skipV:
                            data=align_skels(data)
                            NTUfull.append(list(data))
                            meta.append([S,P,R,A])
                        vid+=1
    NTUfull=np.array(NTUfull)
    # print(NTUfull)
    meta=np.array(meta)
    
    avg = np.average(NTUfull,axis=1)
    dis=5
    for i in range(len(avg)):
        for j in range(len(avg[i])-1):
            avg[i][j]=np.average(avg[i][j:min(j+dis,len(avg[i])-1)], axis=0)
    if save:
        if rand ==999999:
            np.save("/hpc/group/tdunn/action_data/NTU120AlignedMeta.npy", meta)
            np.save("/hpc/group/tdunn/action_data/NTU120Data.npy", NTUfull)
            np.save("/hpc/group/tdunn/action_data/NTU120Aligned.npy", avg)
        else:
            np.save("/hpc/group/tdunn/action_data/FullDataMeta"+str(rand)+".npy", meta)
            np.save("/hpc/group/tdunn/action_data/FullData"+str(rand)+".npy", NTUfull)
            np.save("/hpc/group/tdunn/action_data/SmoothedData"+str(rand)+".npy", avg)

    return [NTUfull,meta]

def make_dict(NTUfull,meta):
    datadict={}
    for j in range(len(meta)):
        datadict["".join(["%03d"%(i) for i in meta[j]])] = NTUfull[j]
    return datadict

def joint_err(NTUfull, save=False):
    avg=np.average(NTUfull, axis=1)
    view=np.array([NTUfull[:,i]-avg for i in range(3)])
    view=view*view
    view=np.sum(view,axis=0)/3
    for j in range(len(view)):
        view[j]=np.sum(view[j],axis=2)
    view=np.array([np.average(view[i],axis=0) for i in range(len(view))])
    np.shape(view)
    jointavg=np.average(view, axis=0)
    print(jointavg)
    if save:
        np.save("/hpc/group/tdunn/action_data/Errors.npy", view)


# ndiff=0
# ntot=0

def random_NTU_sample(n):
    [NTUfull,meta]=get_aligned_NTU(save=True,rand=n)
    print(np.shape(NTUfull))
    print(np.shape(meta))
    smoothNTU=np.load("/hpc/group/tdunn/action_data/SmoothedData"+str(n)+".npy", allow_pickle=True)
    # print(np.shape(smoothNTU))
    exp_id=np.concatenate([np.full(len(smoothNTU[i]),i) for i in range(len(smoothNTU))])
    # NTUdata=np.load("/hpc/group/tdunn/action_data/FullData.npy", allow_pickle=True)
    # meta=np.load("/hpc/group/tdunn/action_data/FullDataMeta.npy", allow_pickle=True)

    # datadict=makedict(NTUfull,meta)

    with open('NTUmetadatarand.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["scene","person","take","action"])
        writer.writerows(meta)

    smoothNTU=np.swapaxes(np.concatenate(smoothNTU),0,1) # now its joint, frame, coord
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
    for i in range(len(smoothNTU)):
        preddict[joints[i]]=smoothNTU[i]
    matout['predictions']=preddict
    print(np.shape(smoothNTU))
    scipy.io.savemat("HumanNTUrand.mat",matout)

def NTU_to_mat(NTUfull,meta):
    print(np.shape(NTUfull))
    print(np.shape(meta))
    smoothNTU=np.load("/hpc/group/tdunn/action_data/SmoothedData.npy", allow_pickle=True)
    smoothNTU=NTUfull
    # print(np.shape(smoothNTU))
    exp_id=np.concatenate([np.full(len(smoothNTU[i]),i) for i in range(len(smoothNTU))])
    # NTUdata=np.load("/hpc/group/tdunn/action_data/FullData.npy", allow_pickle=True)
    # meta=np.load("/hpc/group/tdunn/action_data/FullDataMeta.npy", allow_pickle=True)

    # datadict=makedict(NTUfull,meta)

    with open('NTU60ViewFullMeta.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["scene","person","take","view","action"])
        writer.writerows(meta)

    smoothNTU=np.swapaxes(np.concatenate(smoothNTU),0,1) # now its joint, frame, coord
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
    for i in range(len(smoothNTU)):
        preddict[joints[i]]=smoothNTU[i]
    matout['predictions']=preddict
    print(np.shape(smoothNTU))
    scipy.io.savemat("NTU60ViewFull.mat",matout)

#37874 videos total
# but ignore videos [18119,18222,18446,18562,23367,26246]
# NTU_to_mat()

# get_aligned_NTU(save=True)
# avg=np.load("/hpc/group/tdunn/action_data/SmoothedDataTemp.npy", allow_pickle=True)

# [ntu,meta]=get_raw_NTU()
# ntu=np.load("/hpc/group/tdunn/action_data/NTU120Views.npy", allow_pickle=True)
# meta=np.load("/hpc/group/tdunn/action_data/NTU120MetaViews.npy", allow_pickle=True)
# NTU_to_mat(ntu,meta)
[meta,ntu,aligned]=get_aligned_NTU()
import pdb;pdb.set_trace();

# print(pd.__version__)
# NTU_plot_files([["S001C001P001R002A007","S001C002P001R002A007","S001C003P001R002A007"]],name="apr5_2.mp4",rgb=True, avgsmooth=False)

# jointErr(NTUfull)
# import pdb;pdb.set_trace();

# jointerr=[[[] for k in range(25)] for l in range(120)]
# for i in range(len(NTUdata)):
# # for i in range(1):
#     for j in range(25):
#         # if i>=18707:
#         #     import pdb;pdb.set_trace();
#         v12=3*(np.array([NTUdata[i][0][k][j] for k in range(len(NTUdata[i][0]))])-np.array([NTUdata[i][1][k][j] for k in range(len(NTUdata[i][1]))]))**2
#         v23=3*(np.array([NTUdata[i][1][k][j] for k in range(len(NTUdata[i][1]))])-np.array([NTUdata[i][2][k][j] for k in range(len(NTUdata[i][2]))]))**2
#         v13=3*(np.array([NTUdata[i][0][k][j] for k in range(len(NTUdata[i][0]))])-np.array([NTUdata[i][2][k][j] for k in range(len(NTUdata[i][2]))]))**2
#         jointerr[int(meta[i][3])-1][j].append(((v12.mean())**0.5+(v23.mean())**0.5+(v13.mean())**0.5)/3)
#     print(i)
# np.save("/hpc/group/tdunn/hk276/CAPTURE_demo/Python_analysis/engine/test/jointerr.npy", jointerr)


# jointerr=np.load("/hpc/group/tdunn/hk276/CAPTURE_demo/Python_analysis/engine/test/jointerr.npy", allow_pickle=True)

# print([[np.average(jointerr[i][j]) for j in range(25)] for i in range(len(jointerr))])


# allframes=[[] for j in range(60)]
# for i in range(len(NTUdata)):
#     allframes[meta[i][3]-1].append(len(NTUdata[i][0]))
# avgframes=[np.average(i) for i in allframes]



# filenames=[]
# for i in range(1,10):
#     filenames.append("S001C003P001R001A00"+str(i))
# filenames.append("S001C001P001R001A007")
# filenames.append("S001C002P001R001A007")
# filenames.append("S001C003P001R001A007")

# # added afterwards
# errs=[1.85492038, 2.12548352, 1.06897217, 0.978917, 0.70836396, 1.4322373, 2.48995813, 2.94517612, 1.85898066, 1.13448, 1.34619541, 0.91291215, 1.34257265, 5.19916137, 1.1336672, 1.49055882, 1.40160912, 0.86869816, 0.79155459, 1.88988937, 1.06147074, 1.10175126, 1.04357012, 1.84837574, 0.60736527, 0.66856325, 0.69025604, 0.74275938, 0.54954149, 0.75252743, 1.04403935, 0.95417155, 0.63454808, 0.62278359, 1.18504817, 0.76199065, 1.59846476, 0.89933538, 0.91067872, 0.67566878, 0.8459668, 1.71047651, 2.09132947, 0.86739868, 0.79116921, 0.93934046, 0.96809681, 1.26941141, 1.00359966, 2.71052344, 1.47015487, 2.68639048, 1.44213265, 1.39372011, 9.87018262, 1.43459341, 1.75457397, 1.21038197, 18.60886659, 5.47987644]
# top=np.argsort(errs)
# sortedactions=[act_class[i] for i in top]
# sorterrs=[errs[top[i]] for i in range(len(errs))]

# plt.figure(1,figsize=(20,10))
# plt.bar(sortedactions,sorterrs, color=['red' if top[i] >= 50 else 'black' for i in range(len(top))])
# plt.title("Action Classes with Highest Cross-view Agreement")
# plt.ylabel("Error")
# plt.grid(True)
# for i in range(len(sorterrs)):
#     plt.text(i,errs[top[i]]+0.1,top[i],ha='center',fontsize=12)
# #plt.ylim(0,1)
# plt.xticks(rotation='vertical')
# plt.show()




# # Calculating velocities and standard deviation of velocites over windows
# abs_vel = get_velocities(pose, 
#                          pose_struct.exp_ids_full, 
#                          pose_struct.connectivity.joint_names)

# pose = center_spine(pose)
# pose = rotate_spine(pose)
# # vis.skeleton_vid3D(pose,
# #                    pose_struct.connectivity,
# #                    frames=[1000],
# #                    N_FRAMES = 300,
# #                    VID_NAME='vid_centered.mp4',
# #                    SAVE_ROOT='./')

# euclid_vec = get_ego_pose(pose,
#                           pose_struct.connectivity.joint_names)

# angles = get_angles(pose,
#                     pose_struct.connectivity.angles)

# ang_vel = get_angular_vel(angles,
#                           pose_struct.exp_ids_full)

# # head_angv = get_head_angular(pose, pose_struct.exp_ids_full)

# ipca = IncrementalPCA(n_components=10, batch_size=100)
# import pdb; pdb.set_trace()
# feat_dict = {
#     'euc_vec': euc_vec,
#     'angles': angles,
#     'abs_vel': abs_vel,
# }
# import pdb; pdb.set_trace()
# pca_feat = {}
# for feat in feat_dict:
#     pca_feat[feat] = ipca.fit_transform(feat_dict[feat])
# # link_lengths = get_lengths(pose,pose_struct.exp_ids_full,pose_struct.connectivity.links)
# # feats = np.concatenate((euc_vec, angles, abs_vel, head_angular),axis=1)

# import pdb; pdb.set_trace()
# #mean center before pca, separate for

# # w_let = wavelet(feats_pca)

# import pdb; pdb.set_trace()