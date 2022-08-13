import os
import os.path
import numpy as np
from numpy.random import randint
import torch
import torch.utils.data as data


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class TSNDataSet(data.Dataset):
    def __init__(self, root_path, list_file, num_dataload,
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='img_{:05d}.t7', transform=None,
                 force_grayscale=False, random_shift=True, test_mode=False, triple=0):

        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.num_dataload = num_dataload
        self.triple = triple

        if self.modality == 'RGBDiff' or self.modality == 'RGBDiff2' or self.modality == 'RGBDiffplus':
            self.new_length += 1

        self._parse_list()

    def _load_feature(self, directory, idx):
        if self.modality == 'RGB' or self.modality == 'RGBDiff' or self.modality == 'RGBDiff2' or self.modality == 'RGBDiffplus':
            feat_path = os.path.join(directory, self.image_tmpl.format(idx))
            try:
                feat = [torch.load(feat_path)]
            except:
                print('Error loading:' % (feat_path))
            return feat

        elif self.modality == 'Flow':
            x_feat = torch.load(os.path.join(
                directory, self.image_tmpl.format('x', idx)))
            y_feat = torch.load(os.path.join(
                directory, self.image_tmpl.format('y', idx)))

            return [x_feat, y_feat]

    def _parse_list(self):
        self.video_list = [VideoRecord(x.strip().split(' '))
                           for x in open(self.list_file)]
        n_repeat = self.num_dataload//len(self.video_list)
        n_left = self.num_dataload % len(self.video_list)
        self.video_list = self.video_list*n_repeat + self.video_list[:n_left]

    def _sample_indices(self, record):
        average_duration = (record.num_frames -
                            self.new_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + \
                randint(average_duration, size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames -
                              self.new_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_val_indices(self, record):
        num_min = self.num_segments + self.new_length - 1
        num_select = record.num_frames - self.new_length + 1

        if record.num_frames >= num_min:
            tick = float(num_select) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * float(x))
                               for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_test_indices(self, record):
        num_min = self.num_segments + self.new_length - 1
        num_select = record.num_frames - self.new_length + 1

        if record.num_frames >= num_min:
            tick = float(num_select) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * float(x))
                               for x in range(self.num_segments)])
        else:
            id_select = np.array([x for x in range(num_select)])
            id_expand = np.ones(self.num_segments-num_select,
                                dtype=int)*id_select[id_select[0]-1]
            offsets = np.append(id_select, id_expand)

        return offsets + 1

    def __getitem__(self, index):
        record = self.video_list[index]

        if not self.test_mode:
            segment_indices = self._sample_indices(
                record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)

        data_ancher, label_ancher = self.get(record, segment_indices)

        if self.triple:
            perm = np.random.permutation(data_ancher.shape[0])
            data_pos = data_ancher[perm]

            index_neg = np.random.randint(self.num_dataload)
            record_neg = self.video_list[index_neg]

            if not self.test_mode:
                segment_indices_neg = self._sample_indices(
                    record_neg) if self.random_shift else self._get_val_indices(record_neg)
            else:
                segment_indices_neg = self._get_test_indices(record_neg)

            data_neg, _ = self.get(record_neg, segment_indices_neg)

            return data_ancher, label_ancher, data_pos, data_neg

        return data_ancher, label_ancher

    def get(self, record, indices):

        frames = list()

        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                seg_feats = self._load_feature(record.path, p)
                frames.extend(seg_feats)

                if p < record.num_frames:
                    p += 1

        process_data = torch.stack(frames)

        return process_data, record.label

    def __len__(self):
        return len(self.video_list)
