
import os
import numpy as np
import cv2

def load_kth(root, freq, strides, current, height, width):
    path = os.path.join(root, 'kth/')
    class_list = os.listdir(path)
    data = list()
    for fname in class_list:
        fpath = path + fname
        flist = list(os.listdir(fpath))
        for video in flist:
            img_path = fpath + '/' + video
            capture = capture_video(img_path, freq, strides, current, height, width)
            data.append(capture)
        print(fname, len(flist), end=' ')
    dataset = np.array(data)
    print()
    print('(samples, frames, w, h, c) :',dataset.shape,'\n')
    return dataset

def capture_video(path, freq, strides, current, height, width):
    vidcap = cv2.VideoCapture(path)
    if current > 0 :
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, current)
    if height != 120 :
        vidcap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if height != 160 :
        vidcap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    frames=[]
    for i in range(freq):
        for j in range(strides):
            success, image = vidcap.read()
        frames.append(image)
    frames = np.array(frames)
    return frames

def video_capture(path, current=0, height=120, width=160):
    vidcap = cv2.VideoCapture(path)
    if current > 0 :
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, current)
    if height != 120 :
        vidcap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if height != 160 :
        vidcap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    
    return vidcap

def print_vidcap_info(vid_cap):
    print('초당 프레임 수',vid_cap.get(cv2.CAP_PROP_FPS))
    print('height',vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print('width',vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print('총 프레임 수 :',vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('현재 프레임 번호 :',vid_cap.get(cv2.CAP_PROP_POS_FRAMES))
    print('노출 :',vid_cap.get(cv2.CAP_PROP_EXPOSURE))
    print('영상 길이 :', vid_cap.get(cv2.CAP_PROP_FRAME_COUNT) / vid_cap.get(cv2.CAP_PROP_FPS),'s' )
    print('프레임 당 시간 간격 :', 1 / vid_cap.get(cv2.CAP_PROP_FPS),'s')
