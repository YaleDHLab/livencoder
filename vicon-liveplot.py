#!/usr/bin/env python

import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d.art3d import juggle_axes

from keras.models import load_model

DEFAULT_POINT_LABELS = ['C7',
          'CLAV', 'LANK',
          'LBHD', 'LBSH',
          'LBWT', 'LELB',
          'LFHD', 'LFRM',
          'LFSH', 'LFWT',
          'LHEL', 'LIEL',
          'LIHAND', 'LIWR',
          'LKNE', 'LKNI',
          'LMT1', 'LMT5',
          'LOHAND', 'LOWR',
          'LSHN', 'LTHI',
          'LTOE', 'LUPA',
          'MBWT',
          'MFWT', 'RANK',
          'RBHD', 'RBSH',
          'RBWT', 'RELB',
          'RFHD', 'RFRM',
          'RFSH', 'RFWT',
          'RHEL', 'RIEL',
          'RIHAND', 'RIWR',
          'RKNE', 'RKNI',
          'RMT1', 'RMT5',
          'ROHAND', 'ROWR',
          'RSHN', 'RTHI',
          'RTOE', 'RUPA',
          'STRN',
          'T10']

def update3d(history, fig, ax, ranges, **kwargs):
    ax.cla()
    #ax.set_xlim3d(-2000,2000)
    #ax.set_ylim3d(-2000,2000)
    #ax.set_zlim3d(0,1000)
    ax.set_xlim3d(*ranges[0])
    ax.set_ylim3d(*ranges[1])
    ax.set_zlim3d(*ranges[2])
    alphas = np.linspace(0.1,1,len(history))**2
    for x,a in zip(history, alphas):
        ax.scatter(x[:,0],x[:,1],x[:,2], alpha=a, **kwargs)

def update2d(history, fig, ax, ranges, **kwargs):
    ax.cla()
    ax.set_xlim(*ranges[0])
    ax.set_ylim(*ranges[1])
    alphas = np.linspace(0.1,1,len(history))**2
    for x,a in zip(history, alphas):
        ax.scatter(x[:,0], x[:,1], alpha=a, **kwargs)

if __name__ == "__main__":
    n_skip = 4

    current_frame = None
    current_subject_id = 0
    plt.ion()

    fig = plt.figure(figsize=(21,7))
    ax1 = fig.add_subplot(131, projection='3d')

    #ax.set_xlim3d(-500,500)
    #ax.set_ylim3d(400,1400)
    #ax.set_zlim3d(0,1000)
    #ax.hold(True)

    ax1_ranges = [(-0.5,0.5),(-0.5,0.5),(0,0.5)]

    #ax2 = fig.add_subplot(132, projection='3d')
    ax2 = fig.add_subplot(132)
    ax2_ranges = [(-2,2),(-2,2),(-2,2)]

    ax3 = fig.add_subplot(133)

    ax3_ranges = [(-1.2,1.2),(-1.2,1.2)]

    encoder = load_model('pose-enc-raymond.h5')
    encoder.summary()

    iframe = 0
    points = None
    history_x = []
    history_z = []
    history_w = []
    h_len_x = 4
    h_len_z = 4*32
    h_len_w = 4*32

    larm_history = []
    rarm_history = []
    while True:
        l = sys.stdin.readline()
        if not l:
            break
        l = l.strip()
        if not ':' in l: continue
        items = l.split(':')
        if not len(items) == 2: continue
        k, v = items
        if k == 'frame_number':
            current_frame = dict(frame_number=int(v))
            continue

        if current_frame is None:
            continue

        if k == 'n_subjects':
            current_frame['n_subjects'] = int(v)
            current_frame['subjects'] = [{} for _ in range(current_frame['n_subjects'])]
            #assert current_frame['n_subjects'] == 1
        elif k == 'subject_id':
            current_subject_id = int(v)
        elif k == 'subject_name':
            current_frame['subjects'][current_subject_id]['subject_name'] = v
        elif k == 'markers':
            current_frame['subjects'][current_subject_id]['markers'] = v.split(',')
        elif k == 'coords':
            current_frame['subjects'][current_subject_id]['coords'] = np.float_(v.split(',')).reshape((-1,3))
        elif l.startswith('end_frame:'):
            x = current_frame['subjects'][0]['coords']
            #print(x.shape)
            #print(current_frame)
            iframe += 1
            markers = current_frame['subjects'][0]['markers']
            lelb = markers.index('LELB')
            lowr = markers.index('LOWR')
            relb = markers.index('RELB')
            rowr = markers.index('ROWR')
            larm = np.sqrt(np.sum((x[lelb]-x[lowr])**2,axis=-1))
            rarm = np.sqrt(np.sum((x[relb]-x[rowr])**2,axis=-1))
            larm_history.append(larm)
            rarm_history.append(rarm)
            armlen = 0.5*(np.median(larm) + np.median(rarm))
            print("arm =",armlen)
            if iframe % n_skip == 0:
                ordering = [DEFAULT_POINT_LABELS.index(m) for m in current_frame['subjects'][0]['markers']]
                x_prep = x[ordering] / armlen / 10.
                x_prep[:,:2] -= x_prep[:,:2].mean(axis=-2, keepdims=True)
                z,w = encoder.predict(np.expand_dims(x_prep[ordering],axis=0))
                #w = np.array([np.cos(w), np.sin(w)])
                print(z.shape)
                history_x.append(x_prep)
                history_x = history_x[-h_len_x:]
                #history_z.append(z)
                history_z.append(z[:,:2])
                history_z = history_z[-h_len_z:]
                history_w.append(w)
                history_w = history_w[-h_len_w:]
                update3d(history_x, fig, ax1, ax1_ranges, color='C0',s=10)
                #update3d(history_z, fig, ax2, ax2_ranges, color='C1',s=70)
                update2d(history_z, fig, ax2, ax2_ranges, color='C1',s=70)
                update2d(history_w, fig, ax3, ax3_ranges, color='C2',s=300)
                print(w)
                plt.draw()
                plt.pause(0.0005)
