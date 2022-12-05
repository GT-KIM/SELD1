#
# Data generator for training the SELDnet
#

import os
import numpy as np
import scripts.cls_feature_class as cls_feature_class
from IPython import embed
from collections import deque
import random


def time_mixing(feat, label):
    label_idx = list(np.arange(len(label)))
    random.shuffle(label_idx)
    feat_idx = list(np.zeros(len(feat), dtype=int))
    for j in range(len(label)):
        feat_idx[5 * j]= 5 * label_idx[j]
        feat_idx[5 * j + 1]= 5 * label_idx[j] + 1
        feat_idx[5 * j + 2]= 5 * label_idx[j] + 2
        feat_idx[5 * j + 3]= 5 * label_idx[j] + 3
        feat_idx[5 * j + 4]= 5 * label_idx[j] + 4

    feat = np.array(feat)
    label = np.array(label)

    feat = feat[feat_idx]
    label = label[label_idx]

    #feat = deque(feat)
    #label = deque(label)

    return feat, label
class DataGenerator(object):
    def __init__(
            self, params, split=1, shuffle=True, per_file=False, is_eval=False, is_test=False, is_test2=False
    ):
        self.params = params
        self._is_test = is_test
        self._is_test2 = is_test2
        self._per_file = per_file
        self._is_eval = is_eval
        self._splits = np.array(split)
        self._batch_size = params['batch_size']
        self._feature_seq_len = params['feature_sequence_length']
        self._label_seq_len = params['label_sequence_length']
        self._shuffle = shuffle
        self._feat_cls = cls_feature_class.FeatureClass(params=params, is_eval=self._is_eval)
        self._label_dir = self._feat_cls.get_label_dir()
        self._feat_dir = self._feat_cls.get_normalized_feat_dir()
        self._multi_accdoa = params['multi_accdoa']

        self._filenames_list = list()
        self._nb_frames_file = 0  # Using a fixed number of frames in feat files. Updated in _get_label_filenames_sizes()
        self._nb_mel_bins = self._feat_cls.get_nb_mel_bins()
        self._nb_ch = None
        self._label_len = None  # total length of label - DOA + SED
        self._doa_len = None  # DOA label length
        self._nb_classes = self._feat_cls.get_nb_classes()

        self._circ_buf_feat = None
        self._circ_buf_label = None

        self._get_filenames_list_and_feat_label_sizes()

        print(
            '\tDatagen_mode: {}, nb_files: {}, nb_classes:{}\n'
            '\tnb_frames_file: {}, feat_len: {}, nb_ch: {}, label_len:{}\n'.format(
                'eval' if self._is_eval else 'dev', len(self._filenames_list), self._nb_classes,
                self._nb_frames_file, self._nb_mel_bins, self._nb_ch, self._label_len
            )
        )

        print(
            '\tDataset: {}, split: {}\n'
            '\tbatch_size: {}, feat_seq_len: {}, label_seq_len: {}, shuffle: {}\n'
            '\tTotal batches in dataset: {}\n'
            '\tlabel_dir: {}\n '
            '\tfeat_dir: {}\n'.format(
                params['dataset'], split,
                self._batch_size, self._feature_seq_len, self._label_seq_len, self._shuffle,
                self._nb_total_batches,
                self._label_dir, self._feat_dir
            )
        )

    def get_data_sizes(self):
        feat_shape = (self._batch_size, 7, 200, self._nb_mel_bins)
        if self._is_eval:
            label_shape = None
        else:
            if self._multi_accdoa is True:
                label_shape = (self._batch_size, self._label_seq_len, self._nb_classes * 3 * 3)
            else:
                label_shape = (self._batch_size, self._label_seq_len, self._nb_classes * 3)
        return feat_shape, label_shape

    def get_total_batches_in_data(self):
        return self._nb_total_batches

    def _get_filenames_list_and_feat_label_sizes(self):
        print('Computing some stats about the dataset')
        max_frames, total_frames, temp_feat = -1, 0, []
        i = 0
        for filename in os.listdir(self._feat_dir):
            if int(filename.split('_')[0][4:]) in self._splits:  # check which split the file belongs to
                if self.params['quick_test'] and i > 47 :
                    break
                self._filenames_list.append(filename)
                i += 1
                if os.path.exists('./config/stats.npz'):
                    if self._is_test:
                        temp_feat = np.load(os.path.join(self._feat_dir, filename))
                        total_frames += (temp_feat.shape[0] - (temp_feat.shape[0] % self._feature_seq_len))
                        if temp_feat.shape[0] > max_frames:
                            max_frames = temp_feat.shape[0]
                else:
                    temp_feat = np.load(os.path.join(self._feat_dir, filename))
                    total_frames += (temp_feat.shape[0] - (temp_feat.shape[0] % self._feature_seq_len))
                    if temp_feat.shape[0] > max_frames:
                        max_frames = temp_feat.shape[0]
        if os.path.exists('./config/stats.npz'):
            if self._is_test:
                self._nb_frames_file = max_frames if self._per_file else temp_feat.shape[0]
                self._nb_ch = temp_feat.shape[1]
        else:
            self._nb_frames_file = max_frames if self._per_file else temp_feat.shape[0]
            self._nb_ch = temp_feat.shape[1]

        if not self._is_test:
            if not os.path.exists('./config/stats.npz'):
                np.savez('./config/stats.npz', _filenames_list=self._filenames_list,
                         _nb_frames_file=self._nb_frames_file,
                         _nb_ch=self._nb_ch, total_frames=total_frames, max_frames=max_frames)
            else:
                d = np.load('./config/stats.npz', allow_pickle=True)
                total_frames = d['total_frames']
                max_frames = d['max_frames']
                self._nb_frames_file = int(d['_nb_frames_file'])
                self._nb_ch = int(d['_nb_ch'])

        if not self._is_eval:
            temp_label = np.load(os.path.join(self._label_dir, self._filenames_list[0]))
            if self._multi_accdoa is True:
                self._num_track_dummy = temp_label.shape[-3]
                self._num_axis = temp_label.shape[-2]
                self._num_class = temp_label.shape[-1]
            else:
                self._label_len = temp_label.shape[-1]
            self._doa_len = 3  # Cartesian

        if self._per_file:
            self._batch_size = int(np.ceil(max_frames / float(self._feature_seq_len)))
            print(
                '\tWARNING: Resetting batch size to {}. To accommodate the inference of longest file of {} frames in a single batch'.format(
                    self._batch_size, max_frames))
            self._nb_total_batches = len(self._filenames_list)
        else:
            self._nb_total_batches = int(np.floor(total_frames / (self._batch_size * self._feature_seq_len)))

        self._feature_batch_seq_len = self._batch_size * self._feature_seq_len
        self._label_batch_seq_len = self._batch_size * self._label_seq_len
        return

    def generate(self):
        """
        Generates batches of samples
        :return:
        """
        if self._shuffle:
            random.shuffle(self._filenames_list)

        # Ideally this should have been outside the while loop. But while generating the test data we want the data
        # to be the same exactly for all epoch's hence we keep it here.
        self._circ_buf_feat = deque()
        self._circ_buf_label = deque()

        file_cnt = 0
        if self._is_eval:
            for i in range(self._nb_total_batches):
                # load feat and label to circular buffer. Always maintain atleast one batch worth feat and label in the
                # circular buffer. If not keep refilling it.
                while len(self._circ_buf_feat) < self._feature_batch_seq_len:
                    temp_feat = np.load(os.path.join(self._feat_dir, self._filenames_list[file_cnt]))

                    for row_cnt, row in enumerate(temp_feat):
                        self._circ_buf_feat.append(row)

                    # If self._per_file is True, this returns the sequences belonging to a single audio recording
                    if self._per_file:
                        extra_frames = self._feature_batch_seq_len - temp_feat.shape[0]
                        extra_feat = np.ones((extra_frames, temp_feat.shape[1])) * 1e-6

                        for row_cnt, row in enumerate(extra_feat):
                            self._circ_buf_feat.append(row)

                    file_cnt = file_cnt + 1

                # Read one batch size from the circular buffer
                feat = np.zeros((self._feature_batch_seq_len, self._nb_mel_bins * self._nb_ch))
                for j in range(self._feature_batch_seq_len):
                    feat[j, :] = self._circ_buf_feat.popleft()
                feat = np.reshape(feat, (self._feature_batch_seq_len, self._nb_ch, self._nb_mel_bins))

                # Split to sequences
                feat = self._split_in_seqs(feat, self._feature_seq_len)
                feat = np.transpose(feat, (0, 2, 1, 3))

                yield feat

        elif self._is_test2 :

            for i in range(self._nb_total_batches):

                # load feat and label to circular buffer. Always maintain atleast one batch worth feat and label in the
                # circular buffer. If not keep refilling it.

                temp_feat = np.load(os.path.join(self._feat_dir, self._filenames_list[file_cnt]))
                temp_label = np.load(os.path.join(self._label_dir, self._filenames_list[file_cnt]))

                file_cnt = file_cnt + 1

                idx = 0
                full_feat = list()
                full_label = list()
                label_length = len(temp_label)
                while idx <= len(temp_label) - 50 :
                    curr_feat = temp_feat[idx*5 : (idx+50) * 5, :]
                    curr_label = temp_label[idx : idx+50]

                    full_feat.append(curr_feat)
                    full_label.append(curr_label)

                    idx += 1
                feat = np.array(full_feat)
                label = np.array(full_label)

                mel = feat[:, :, :128 * 4]
                IV = feat[:, :, 128 * 4: 128 * 7]

                mel = np.reshape(mel, (-1, 250, 4, self._nb_mel_bins))
                IV = np.reshape(IV, (-1, 250, 3, self._nb_mel_bins))
                feat = np.concatenate([mel, IV], axis=2)
                feat = feat.transpose((0, 2, 1, 3))
                yield feat, label
        elif self._is_test :

            for i in range(self._nb_total_batches):

                # load feat and label to circular buffer. Always maintain atleast one batch worth feat and label in the
                # circular buffer. If not keep refilling it.

                while len(self._circ_buf_feat) < self._feature_batch_seq_len:
                    temp_feat = np.load(os.path.join(self._feat_dir, self._filenames_list[file_cnt]))
                    temp_label = np.load(os.path.join(self._label_dir, self._filenames_list[file_cnt]))

                    if not self._per_file:
                        # Inorder to support variable length features, and labels of different resolution.
                        # We remove all frames in features and labels matrix that are outside
                        # the multiple of self._label_seq_len and self._feature_seq_len. Further we do this only in training.
                        temp_label = temp_label[:temp_label.shape[0] - (temp_label.shape[0] % self._label_seq_len)]
                        temp_mul = temp_label.shape[0] // self._label_seq_len
                        temp_feat = temp_feat[:temp_mul * self._feature_seq_len, :]

                    for f_row in temp_feat:
                        self._circ_buf_feat.append(f_row)
                    for l_row in temp_label:
                        self._circ_buf_label.append(l_row)

                    # If self._per_file is True, this returns the sequences belonging to a single audio recording
                    if self._per_file:
                        feat_extra_frames = self._feature_batch_seq_len - temp_feat.shape[0]
                        extra_feat = np.ones((feat_extra_frames, temp_feat.shape[1])) * 1e-6

                        label_extra_frames = self._label_batch_seq_len - temp_label.shape[0]
                        if self._multi_accdoa is True:
                            extra_labels = np.zeros(
                                (label_extra_frames, self._num_track_dummy, self._num_axis, self._num_class))
                        else:
                            extra_labels = np.zeros((label_extra_frames, temp_label.shape[1]))

                        for f_row in extra_feat:
                            self._circ_buf_feat.append(f_row)
                        for l_row in extra_labels:
                            self._circ_buf_label.append(l_row)

                    file_cnt = file_cnt + 1

                # Read one batch size from the circular buffer
                feat = np.zeros((self._feature_batch_seq_len, self._nb_ch))
                for j in range(self._feature_batch_seq_len):
                    feat[j, :] = self._circ_buf_feat.popleft()
                mel = feat[:, :128 * 4]
                IV = feat[:, 128 * 4: 128 * 7]
                # IPD = feat[:, 128 * 7:]

                mel = np.reshape(mel, (self._feature_batch_seq_len, 4, self._nb_mel_bins))
                IV = np.reshape(IV, (self._feature_batch_seq_len, 3, self._nb_mel_bins))
                # IPD = np.reshape(IPD, (self._feature_batch_seq_len, -1, 6))
                # IPD = np.transpose(IPD, (0, 2, 1))
                feat = np.concatenate([mel, IV], axis=1)

                if self._multi_accdoa is True:
                    label = np.zeros(
                        (self._label_batch_seq_len, self._num_track_dummy, self._num_axis, self._num_class))
                    for j in range(self._label_batch_seq_len):
                        label[j, :, :, :] = self._circ_buf_label.popleft()
                else:
                    label = np.zeros((self._label_batch_seq_len, self._label_len))
                    for j in range(self._label_batch_seq_len):
                        label[j, :] = self._circ_buf_label.popleft()
                # Split to sequences
                feat = self._split_in_seqs(feat, self._feature_seq_len)
                feat = np.transpose(feat, (0, 2, 1, 3))

                label = self._split_in_seqs(label, self._label_seq_len)
                if self._multi_accdoa is True:
                    pass
                else:
                    mask = label[:, :, :self._nb_classes]
                    mask = np.tile(mask, 3)
                    label = mask * label[:, :, self._nb_classes:]

                yield feat, label
        else :
            temp_feat = list()
            temp_label = list()
            curr_file_idx = np.random.randint(0, len(self._filenames_list), self._batch_size)
            for i in range(self._batch_size):
                temp_feat.append(
                    np.load(os.path.join(self._feat_dir, self._filenames_list[curr_file_idx[i]])))
                temp_label.append(
                    np.load(os.path.join(self._label_dir, self._filenames_list[curr_file_idx[i]])))

            #for i in range(self._batch_size) :
            #    temp_feat[i], temp_label[i] = time_mixing(temp_feat[i], temp_label[i])

            feat_idxs = np.zeros(self._batch_size, dtype=int)

            while True :
                for i in range(self._batch_size) :
                    if len(temp_feat[i]) - feat_idxs[i] < self._feature_seq_len :
                        curr_file_idx[i] = np.random.randint(0, len(self._filenames_list), 1)
                        feat_idxs[i] = 0
                        temp_feat[i] = np.load(os.path.join(self._feat_dir, self._filenames_list[curr_file_idx[i]]))
                        temp_label[i] = np.load(os.path.join(self._label_dir, self._filenames_list[curr_file_idx[i]]))

                feat = np.zeros((self._batch_size, self._feature_seq_len, self._nb_ch))
                label = np.zeros(
                    (self._batch_size, self._label_seq_len, self._num_track_dummy, self._num_axis, self._num_class))
                for i in range(self._batch_size) :
                    feat[i] = temp_feat[i][feat_idxs[i]:feat_idxs[i]+self._feature_seq_len]
                    label[i] = temp_label[i][int(feat_idxs[i] / 5):int(feat_idxs[i] / 5) + self._label_seq_len]
                    feat_idxs[i] += 50
                mel = feat[:, :, :128 * 4]
                IV = feat[:, :, 128 * 4: 128 * 7]

                mel = np.reshape(mel, (self._batch_size, self._feature_seq_len, 4, self._nb_mel_bins))
                IV = np.reshape(IV, (self._batch_size, self._feature_seq_len, 3, self._nb_mel_bins))
                feat = np.concatenate([mel, IV], axis=2)
                feat = np.transpose(feat, (0, 2, 1, 3))

                yield feat, label

    def _split_in_seqs(self, data, _seq_len):
        if len(data.shape) == 1:
            if data.shape[0] % _seq_len:
                data = data[:-(data.shape[0] % _seq_len), :]
            data = data.reshape((data.shape[0] // _seq_len, _seq_len, 1))
        elif len(data.shape) == 2:
            if data.shape[0] % _seq_len:
                data = data[:-(data.shape[0] % _seq_len), :]
            data = data.reshape((data.shape[0] // _seq_len, _seq_len, data.shape[1]))
        elif len(data.shape) == 3:
            if data.shape[0] % _seq_len:
                data = data[:-(data.shape[0] % _seq_len), :, :]
            data = data.reshape((data.shape[0] // _seq_len, _seq_len, data.shape[1], data.shape[2]))
        elif len(data.shape) == 4:  # for multi-ACCDOA with ADPIT
            if data.shape[0] % _seq_len:
                data = data[:-(data.shape[0] % _seq_len), :, :, :]
            data = data.reshape((data.shape[0] // _seq_len, _seq_len, data.shape[1], data.shape[2], data.shape[3]))
        else:
            print('ERROR: Unknown data dimensions: {}'.format(data.shape))
            exit()
        return data

    @staticmethod
    def split_multi_channels(data, num_channels):
        tmp = None
        in_shape = data.shape
        if len(in_shape) == 3:
            hop = in_shape[2] / num_channels
            tmp = np.zeros((in_shape[0], num_channels, in_shape[1], hop))
            for i in range(num_channels):
                tmp[:, i, :, :] = data[:, :, i * hop:(i + 1) * hop]
        elif len(in_shape) == 4 and num_channels == 1:
            tmp = np.zeros((in_shape[0], 1, in_shape[1], in_shape[2], in_shape[3]))
            tmp[:, 0, :, :, :] = data
        else:
            print('ERROR: The input should be a 3D matrix but it seems to have dimensions: {}'.format(in_shape))
            exit()
        return tmp

    def get_nb_classes(self):
        return self._nb_classes

    def nb_frames_1s(self):
        return self._feat_cls.nb_frames_1s()

    def get_hop_len_sec(self):
        return self._feat_cls.get_hop_len_sec()

    def get_filelist(self):
        return self._filenames_list

    def get_frame_per_file(self):
        return self._label_batch_seq_len

    def get_nb_frames(self):
        return self._feat_cls.get_nb_frames()

    def get_data_gen_mode(self):
        return self._is_eval

    def write_output_format_file(self, _out_file, _out_dict):
        return self._feat_cls.write_output_format_file(_out_file, _out_dict)
