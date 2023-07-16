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

conn=ds.Connectivity().load(skeleton_path='/hpc/group/tdunn/hk276/CAPTURE_demo/CAPTURE_data/skeletons.py', skeleton_name='NUCLAhuman')
#act_class=["drink water", "eat meal/snack", "brushing teeth", "brushing hair", "drop", "pickup", "throw", "sitting down", "standing up", "clapping", "reading", "writing", "tear up paper", "wear jacket", "take off jacket", "wear a shoe", "take off a shoe", "wear on glasses", "take off glasses", "put on a hat/cap", "take off a hat/cap", "cheer up", "hand waving", "kicking something", "put something inside pocket", "hopping (one foot jumping)", "jump up", "make a phone call/answer phone", "playing with phone/tablet", "typing on a keyboard", "pointing to something with finger", "taking a selfie", "check time (from watch)", "rub two hands together", "nod head/bow", "shake head", "wipe face", "salute", "put the palms together", "cross hands in front (say stop)", "sneeze/cough", "staggering", "falling", "touch head (headache)", "touch chest (stomachache/heart pain)", "touch back (backache)", "touch neck (neckache)", "nausea or vomiting condition", "use a fan (with hand or paper)/feeling warm", "punching/slapping other person", "kicking other person", "pushing other person", "pat on back of other person", "point finger at the other person", "hugging other person", "giving something to other person", "touch other persons pocket", "handshaking", "walking towards each other", "walking apart from each other"]
act_class=["drink water","eat meal/snack","brushing teeth","brushing hair","drop","pickup","throw","sitting down","standing up (from sitting position)","clapping","reading","writing","tear up paper","wear jacket","take off jacket","wear a shoe","take off a shoe","wear on glasses","take off glasses","put on a hat/cap","take off a hat/cap","cheer up","hand waving","kicking something","reach into pocket","hopping (one foot jumping)","jump up","make a phone call/answer phone","playing with phone/tablet","typing on a keyboard","pointing to something with finger","taking a selfie","check time (from watch)","rub two hands together","nod head/bow","shake head","wipe face","salute","put the palms together","cross hands in front (say stop)","sneeze/cough","staggering","falling","touch head (headache)","touch chest (stomachache/heart pain)","touch back (backache)","touch neck (neckache)","nausea or vomiting condition","use a fan (with hand or paper)/feeling warm","punching/slapping other person","kicking other person","pushing other person","pat on back of other person","point finger at the other person","hugging other person","giving something to other person","touch other person\'s pocket","handshaking","walking towards each other","walking apart from each other","put on headphone","take off headphone","shoot at the basket","bounce ball","tennis bat swing","juggling table tennis balls","hush (quite)","flick hair","thumb up","thumb down","make ok sign","make victory sign","staple book","counting money","cutting nails","cutting paper (using scissors)","snapping fingers","open bottle","sniff (smell)","squat down","toss a coin","fold paper","ball up paper","play magic cube","apply cream on face","apply cream on hand back","put on bag","take off bag","put something into a bag","take something out of a bag","open a box","move heavy objects","shake fist","throw up cap/hat","hands up (both hands)","cross arms","arm circles","arm swings","running on the spot","butt kicks (kick backward)","cross toe touch","side kick","yawn","stretch oneself","blow nose","hit other person with something","wield knife towards other person","knock over other person (hit with body)","grab other person’s stuff","shoot at other person with a gun","step on foot","high-five","cheers and drink","carry something with other person","take a photo of other person","follow other person","whisper in other person’s ear","exchange things with other person","support somebody with hand","finger-guessing game (playing rock-paper-scissors)"]
path_to_skels='/hpc/group/tdunn/action_data/NUCLAskeletonsNP/' 
path_to_vids='/hpc/group/tdunn/action_data/NUCLArgb+d_videos/NUCLArgbd_rgb_s'
# skipA = []
skipV=[]
# skipA = [i for i in range(1,121) if i>29]  
skipA = []

def NUCLA_plot_files(file_names: List = [], name: str="temp.mp4", rgb: bool=False, avgsmooth: bool=False):
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


def align_skels(data: List=[]):
    n=5
    minlen=min([len(data[i]) for i in range(len(data))])
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

def get_raw_NUCLA():
    vid=0
    NUCLAfull=[]
    meta=[]
    for A in range(1,17): #loop across action
        print("############### starting A = "+str(A))
        if A in skipA:
            continue
        for S in range(1,11): #loop across scenes
            print("starting S = "+str(S))
            for E in range(10):
                data=[]
                for C in range(3): # loop across cameras
                    try:
                        f=open('/hpc/group/tdunn/action_data/NUCLA/all_sqe/a%02d_s%02d_e%02d_v%02d.json'%(A,S,E,C+1))
                    except:
                        continue
                    NUCLAfull.append(np.array(json.load(f)['skeletons']))
                    f.close()
                    meta.append([A,S,E,C])
    NUCLAfull=np.array(NUCLAfull)
    print(np.shape(NUCLAfull))
    meta=np.array(meta)
    
    np.save("/hpc/group/tdunn/action_data/NUCLAdatameta.npy", meta)
    np.save("/hpc/group/tdunn/action_data/NUCLAdata.npy", NUCLAfull)

    return [NUCLAfull,meta]

def make_dict(NUCLAfull,meta):
    datadict={}
    for j in range(len(meta)):
        datadict["".join(["%03d"%(i) for i in meta[j]])] = NUCLAfull[j]
    return datadict

def joint_err(NUCLAfull, save=False):
    avg=np.average(NUCLAfull, axis=1)
    view=np.array([NUCLAfull[:,i]-avg for i in range(3)])
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


def NUCLA_to_mat(NUCLAfull,meta):
    # smoothNUCLA=np.load("/hpc/group/tdunn/action_data/SmoothedData.npy", allow_pickle=True)
    smoothNUCLA=NUCLAfull
    # print(np.shape(smoothNUCLA))
    exp_id=np.concatenate([np.full(len(smoothNUCLA[i]),i) for i in range(len(smoothNUCLA))])
    # NUCLAdata=np.load("/hpc/group/tdunn/action_data/FullData.npy", allow_pickle=True)
    # meta=np.load("/hpc/group/tdunn/action_data/FullDataMeta.npy", allow_pickle=True)

    # datadict=makedict(NUCLAfull,meta)

    with open('NUCLAmeta.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["action","person","take","view"])
        writer.writerows(meta)

    smoothNUCLA=np.swapaxes(np.concatenate(smoothNUCLA),0,1) # now its joint, frame, coord
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
    for i in range(len(smoothNUCLA)):
        preddict[joints[i]]=smoothNUCLA[i]
    matout['predictions']=preddict
    print(np.shape(smoothNUCLA))
    scipy.io.savemat("NUCLA.mat",matout)

#37874 videos total
# but ignore videos [18119,18222,18446,18562,23367,26246]
# NUCLA_to_mat()

# get_aligned_NUCLA(save=True)
[NUCLA,meta]=get_raw_NUCLA()
# meta=np.load("/hpc/group/tdunn/action_data/NUCLAdatameta.npy", allow_pickle=True)
# NUCLA=np.load("/hpc/group/tdunn/action_data/NUCLAdata.npy", allow_pickle=True)

# import pdb;pdb.set_trace();
NUCLA_to_mat(NUCLA,meta)
'''
video=[]
frames=sorted(os.listdir("/hpc/group/tdunn/action_data/NUCLA/multiview_action/view_1/a01_s01_e00"))
for i in range(1,len(frames)):
    fdata=[]
    with open("/hpc/group/tdunn/action_data/NUCLA/multiview_action/view_1/a01_s01_e00/"+frames[i]) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            fdata.append(row)
    fdata=np.array(fdata[1:]).astype(float)[:,:-1][:,[0,2,1]]
    video.append(fdata)
video=np.array(video)

conn=ds.Connectivity().load(skeleton_path='/hpc/group/tdunn/hk276/CAPTURE_demo/CAPTURE_data/skeletons.py', skeleton_name='MSRhuman')

vis.skeleton_vid3D(video,
                        connectivity = conn,
                        title="NUCLA test",
                        frames=[int(len(video)/2)],
                        N_FRAMES = len(video)-2,
                        fps=30,
                        VID_NAME="NUCLAskel.mp4",
                        SAVE_ROOT = "./videos/",
                        fsize=25,
                        aspect=[0.9,0.9,1.5],
                        offset=[0.1,0.1])

f=open('/hpc/group/tdunn/action_data/NUCLA/all_sqe/a02_s02_e02_v02.json';len(json.load(f)['skeletons']);f.close()
                        
vis.skeleton_vid3D(NUCLA[0],connectivity = conn,title="NUCLA test",frames=[int(len(NUCLA[0])/2)],N_FRAMES = len(NUCLA[0])-2,fps=30,VID_NAME="NUCLAskel.mp4",SAVE_ROOT = "./videos/",fsize=25,aspect=[0.9,0.9,1.5],offset=[0.1,0.1])
'''

# df=pd.read_csv("/hpc/group/tdunn/action_data/NUCLA/multiview_action/view_1/a01_s01_e00/"+f,delimiter=" ")
# arr=np.reshape(df.values,(-1,41))[:,1:]
# arr=np.char.split(arr.astype(str),sep=' ')
# new=np.zeros((len(arr),20,3))
# for i in range(len(arr)):
#     new[i]=np.array([np.array(arr[i][j])[0:-1].astype(float) for j in range(len(arr[i])) if j%2==0])
# new=new[:,:,[0,2,1]]



# import pdb;pdb.set_trace();
# want it to be video, frame, joint, coordinate

# print(pd.__version__)
# NUCLA_plot_files([["S001C001P001R002A007","S001C002P001R002A007","S001C003P001R002A007"]],name="apr5_2.mp4",rgb=True, avgsmooth=False)

# jointErr(NUCLAfull)
# import pdb;pdb.set_trace();

# jointerr=[[[] for k in range(25)] for l in range(120)]
# for i in range(len(NUCLAdata)):
# # for i in range(1):
#     for j in range(25):
#         # if i>=18707:
#         #     import pdb;pdb.set_trace();
#         v12=3*(np.array([NUCLAdata[i][0][k][j] for k in range(len(NUCLAdata[i][0]))])-np.array([NUCLAdata[i][1][k][j] for k in range(len(NUCLAdata[i][1]))]))**2
#         v23=3*(np.array([NUCLAdata[i][1][k][j] for k in range(len(NUCLAdata[i][1]))])-np.array([NUCLAdata[i][2][k][j] for k in range(len(NUCLAdata[i][2]))]))**2
#         v13=3*(np.array([NUCLAdata[i][0][k][j] for k in range(len(NUCLAdata[i][0]))])-np.array([NUCLAdata[i][2][k][j] for k in range(len(NUCLAdata[i][2]))]))**2
#         jointerr[int(meta[i][3])-1][j].append(((v12.mean())**0.5+(v23.mean())**0.5+(v13.mean())**0.5)/3)
#     print(i)
# np.save("/hpc/group/tdunn/hk276/CAPTURE_demo/Python_analysis/engine/test/jointerr.npy", jointerr)


# jointerr=np.load("/hpc/group/tdunn/hk276/CAPTURE_demo/Python_analysis/engine/test/jointerr.npy", allow_pickle=True)

# print([[np.average(jointerr[i][j]) for j in range(25)] for i in range(len(jointerr))])


# allframes=[[] for j in range(60)]
# for i in range(len(NUCLAdata)):
#     allframes[meta[i][3]-1].append(len(NUCLAdata[i][0]))
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