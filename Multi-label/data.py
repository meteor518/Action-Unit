import os
import pickle
import random

import mxnet as mx
from mxnet import recordio
from mxnet.gluon.data.dataset import Dataset


class ImageRecordDataset(Dataset):
    def __init__(self, filenames, transform=None, with_landmark=False, horizontal_flip=False):
        self.filenames = filenames
        self.transform = transform
        self.with_landmark = with_landmark
        self.horizontal_flip = horizontal_flip
        self._fork()

    def __getitem__(self, idx):
        i, j = self.orig_idx[idx]
        item = self.records[i].read_idx(j)
        header, s = recordio.unpack(item)
        img = mx.image.imdecode(s, 1)
        label = header.label
        if self.transform:
            img, label = self.transform(img, label)
        need_flip = self.horizontal_flip and random.random() > 0.5
        img = mx.nd.flip(img, axis=2) if need_flip else img
        if self.with_landmark:
            landmark = mx.nd.array(self.landmark_dicts[i][j])
            landmark = landmark[17 * 2:-8 * 2]
            landmark = mx.nd.clip(landmark, 0, 224)
            if need_flip:
                landmark[::2] = 224 - landmark[::2]
            return img, label, landmark
        return img, label

    def __len__(self):
        return len(self.orig_idx)

    def _fork(self):
        self.records = [recordio.MXIndexedRecordIO(os.path.splitext(fname)[0] + '.idx', fname, 'r')
                        for fname in self.filenames]
        self.orig_idx = {}
        idx = 0
        for i, r in enumerate(self.records):
            for j in r.keys:
                self.orig_idx[idx] = (i, j)
                idx += 1

        self.landmark_dicts = {}
        if self.with_landmark:
            for i, fname in enumerate(self.filenames):
                landmark_file = os.path.splitext(fname)[0] + '.landmark'
                with open(landmark_file, 'rb') as f:
                    self.landmark_dicts[i] = pickle.load(f)
