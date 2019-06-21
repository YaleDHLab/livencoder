#!/usr/bin/env python

import numpy as np
import time

sleep = 0.01 # 120 fps sleep

default_labels = [
  'C7', 'CLAV', 'LANK', 'LBHD', 'LBSH',
  'LBWT', 'LELB', 'LFHD', 'LFRM', 'LFSH',
  'LFWT', 'LHEL', 'LIEL', 'LIHAND', 'LIWR',
  'LKNE', 'LKNI', 'LMT1', 'LMT5', 'LOHAND',
  'LOWR', 'LSHN', 'LTHI', 'LTOE', 'LUPA',
  'MBWT', 'MFWT', 'RANK', 'RBHD', 'RBSH',
  'RBWT', 'RELB', 'RFHD', 'RFRM', 'RFSH',
  'RFWT', 'RHEL', 'RIEL', 'RIHAND', 'RIWR',
  'RKNE', 'RKNI', 'RMT1', 'RMT5', 'ROHAND',
  'ROWR', 'RSHN', 'RTHI', 'RTOE', 'RUPA',
  'STRN', 'T10'
]

if __name__ == '__main__':
  in_file = 'raymond-sample.npy'
  d = np.load(in_file)
  d = d[:,np.arange(d.shape[1])[1:]]
  d = d[4200:]

  while True:
    for iframe in range(d.shape[0]):
      time.sleep(sleep)
      print('frame_number:%d'%iframe)
      print('n_subjects:1')
      print('subject_id:0')
      print('subject_name:Raymond')
      print('markers:%s'%','.join(default_labels))
      print('coords:%s'%','.join(['%g'%x for x in d[iframe].reshape(-1)]))
      print('end_frame:%d'%iframe)
      print()
