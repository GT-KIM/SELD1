import os
import scripts.SELD_evaluation_metrics as SELD_evaluation_metrics
import scripts.cls_feature_class as cls_feature_class
import config.parameters as parameters
import numpy as np
from sklearn.cluster import DBSCAN

class ComputeSELDResults(object):
    def __init__(
            self, params, ref_files_folder=None, use_polar_format=True
    ):
        self.params = params
        self._use_polar_format = use_polar_format
        self._datasets = os.listdir(params['dataset_dir'])

        self._desc_dirs = [os.path.join(params['dataset_dir'], dataset, 'metadata') for dataset in self._datasets]
        self._doa_thresh = params['lad_doa_thresh']

        # Load feature class
        self._feat_cls = cls_feature_class.FeatureClass(params)
        
        # collect reference files
        self._ref_labels = {}
        for desc_dir in self._desc_dirs :
            for ref_file in os.listdir(desc_dir):
                # Load reference description file
                gt_dict = self._feat_cls.load_output_format_file(os.path.join(desc_dir, ref_file))
                if not self._use_polar_format:
                    gt_dict = self._feat_cls.convert_output_format_polar_to_cartesian(gt_dict)
                nb_ref_frames = max(list(gt_dict.keys()))
                self._ref_labels[ref_file] = [self._feat_cls.segment_labels(gt_dict, nb_ref_frames), nb_ref_frames]

        self._nb_ref_files = len(self._ref_labels)
        self._average = params['average']

    @staticmethod
    def get_nb_files(file_list, tag='all'):
        '''
        Given the file_list, this function returns a subset of files corresponding to the tag.

        Tags supported
        'all' -
        'ir'

        :param file_list: complete list of predicted files
        :param tag: Supports two tags 'all', 'ir'
        :return: Subset of files according to chosen tag
        '''
        _group_ind = {'room': 10}
        _cnt_dict = {}
        for _filename in file_list:

            if tag == 'all':
                _ind = 0
            else:
                _ind = int(_filename[_group_ind[tag]])

            if _ind not in _cnt_dict:
                _cnt_dict[_ind] = []
            _cnt_dict[_ind].append(_filename)

        return _cnt_dict

    def get_SELD_Results(self, pred_files_path):
        # collect predicted files info
        pred_files = os.listdir(pred_files_path)
        eval = SELD_evaluation_metrics.SELDMetrics(nb_classes=self._feat_cls.get_nb_classes(), doa_threshold=self._doa_thresh, average=self._average)
        for pred_cnt, pred_file in enumerate(pred_files):
            # Load predicted output format file
            pred_dict = self._feat_cls.load_output_format_file(os.path.join(pred_files_path, pred_file))
            if self._use_polar_format:
                pred_dict = self._feat_cls.convert_output_format_cartesian_to_polar(pred_dict, None, is_augment=False)
            pred_labels = self._feat_cls.segment_labels(pred_dict, self._ref_labels[pred_file][1])

            # Calculated scores
            eval.update_seld_scores(pred_labels, self._ref_labels[pred_file][0])

        # Overall SED and DOA scores
        ER, F, LE, LR, seld_scr, classwise_results = eval.compute_seld_scores()

        return ER, F, LE, LR, seld_scr, classwise_results

    def get_SELD_Results_Augmented(self, pred_files_path):
        # collect predicted files info
        pred_files = os.listdir(pred_files_path)
        eval = SELD_evaluation_metrics.SELDMetrics(nb_classes=self._feat_cls.get_nb_classes(), doa_threshold=self._doa_thresh, average=self._average)
        for pred_cnt, pred_file in enumerate(pred_files):
            if pred_file.split('.')[0][-1] == '0' :
                full_pred_dict = list()
                for i in range(8) :
                    curr_pred_file = pred_file.split('.')[0][:-1] + str(i) + '.csv'
                    # Load predicted output format file
                    pred_dict = self._feat_cls.load_output_format_file(os.path.join(pred_files_path, curr_pred_file))
                    if self._use_polar_format:
                        pred_dict = self._feat_cls.convert_output_format_cartesian_to_polar(pred_dict, i, is_augment=True)
                    full_pred_dict.append(pred_dict)
                full_pred_file = pred_file.split('.')[0][:-2] + '.csv'
                pred_labels = self._feat_cls.segment_labels(full_pred_dict[2], self._ref_labels[full_pred_file][1])

                # Calculated scores
                eval.update_seld_scores(pred_labels, self._ref_labels[full_pred_file][0])
            else :
                continue

        # Overall SED and DOA scores
        ER, F, LE, LR, seld_scr, classwise_results = eval.compute_seld_scores()

        return ER, F, LE, LR, seld_scr, classwise_results


    def get_SELD_Results_Augmented_Full(self, pred_files_path):
        # collect predicted files info
        pred_files = os.listdir(pred_files_path)
        eval = SELD_evaluation_metrics.SELDMetrics(nb_classes=self._feat_cls.get_nb_classes(), doa_threshold=self._doa_thresh, average=self._average)
        for pred_cnt, pred_file in enumerate(pred_files):
            if pred_file.split('.')[0][-1] == '0' :
                full_pred_file = pred_file.split('.')[0][:-2] + '.csv'
                for i in range(8) :
                    curr_pred_file = pred_file.split('.')[0][:-1] + str(i) + '.csv'
                    # Load predicted output format file
                    pred_dict = self._feat_cls.load_output_format_file(os.path.join(pred_files_path, curr_pred_file))
                    if self._use_polar_format:
                        pred_dict = self._feat_cls.convert_output_format_cartesian_to_polar(pred_dict, i, is_augment=True)
                    if i == 0 :
                        full_pred_dict = pred_dict
                    else :
                        for j in range(self._ref_labels[full_pred_file][1]) :
                            if j in full_pred_dict and j in pred_dict :
                                full_pred_dict[j] += pred_dict[j]
                            elif not j in full_pred_dict and j in pred_dict :
                                full_pred_dict[j] = pred_dict[j]
                            else :
                                continue
                for j in range(self._ref_labels[full_pred_file][1]):
                    if j in full_pred_dict:
                        curr_frame_result = full_pred_dict[j]
                        X = list()
                        for k in range(len(curr_frame_result)):
                            X.append([curr_frame_result[k][0], curr_frame_result[k][2], curr_frame_result[k][3]])
                        X = np.array(X)
                        clustering1 = X[:, 0]
                        #X1 = X[:, 0, np.newaxis]
                        #clustering1 = DBSCAN(eps=0.5, min_samples=4).fit_predict(X1)
                        X2 = X[:, 1:]
                        clustering2 = DBSCAN(eps=120, min_samples=2).fit_predict(X2)

                        curr_frame_candidate = list()
                        curr_frame_cluster1 = list()
                        curr_frame_cluster2 = list()
                        for k1 in range(len(curr_frame_result)):
                            same_flag = False
                            if clustering1[k1] > -1 and clustering2[k1] > -1 :
                                if k1 == 0:
                                    curr_frame_candidate.append([curr_frame_result[k1]])
                                    curr_frame_cluster1.append(clustering1[k1])
                                    curr_frame_cluster2.append(clustering2[k1])
                                else:
                                    for k2 in range(len(curr_frame_candidate)):
                                        if (clustering1[k1] == curr_frame_cluster1[k2]) and \
                                                (clustering2[k1] == curr_frame_cluster2[k2]):
                                            same_flag = True
                                            curr_frame_candidate[k2].append(curr_frame_result[k1])
                                    if not same_flag:
                                        curr_frame_candidate.append([curr_frame_result[k1]])
                                        curr_frame_cluster1.append(clustering1[k1])
                                        curr_frame_cluster2.append(clustering2[k1])
                        if len(curr_frame_candidate) > 0 :
                            result_list = list()
                            for k2, candidate in enumerate(curr_frame_candidate) :
                                result_list.append(np.mean(candidate, axis=0))
                            full_pred_dict[j] = result_list
                pred_labels = self._feat_cls.segment_labels(full_pred_dict, self._ref_labels[full_pred_file][1])

                # Calculated scores
                eval.update_seld_scores(pred_labels, self._ref_labels[full_pred_file][0])
            else :
                continue

        # Overall SED and DOA scores
        ER, F, LE, LR, seld_scr, classwise_results = eval.compute_seld_scores()

        return ER, F, LE, LR, seld_scr, classwise_results


    def get_consolidated_SELD_results(self, pred_files_path, score_type_list=['all', 'room']):
        '''
            Get all categories of results.
            ;score_type_list: Supported
                'all' - all the predicted files
                'room' - for individual rooms

        '''

        # collect predicted files info
        pred_files = os.listdir(pred_files_path)
        nb_pred_files = len(pred_files)

        # Calculate scores for different splits, overlapping sound events, and impulse responses (reverberant scenes)

        print('Number of predicted files: {}\nNumber of reference files: {}'.format(nb_pred_files, self._nb_ref_files))
        print('\nCalculating {} scores for {}'.format(score_type_list, os.path.basename(pred_output_format_files)))

        for score_type in score_type_list:
            print('\n\n---------------------------------------------------------------------------------------------------')
            print('------------------------------------  {}   ---------------------------------------------'.format('Total score' if score_type=='all' else 'score per {}'.format(score_type)))
            print('---------------------------------------------------------------------------------------------------')

            split_cnt_dict = self.get_nb_files(pred_files, tag=score_type) # collect files corresponding to score_type
            # Calculate scores across files for a given score_type
            for split_key in np.sort(list(split_cnt_dict)):
                # Load evaluation metric class
                eval = SELD_evaluation_metrics.SELDMetrics(nb_classes=self._feat_cls.get_nb_classes(), doa_threshold=self._doa_thresh, average=self._average)
                for pred_cnt, pred_file in enumerate(split_cnt_dict[split_key]):
                    # Load predicted output format file
                    pred_dict = self._feat_cls.load_output_format_file(os.path.join(pred_output_format_files, pred_file))
                    if self._use_polar_format:
                        pred_dict = self._feat_cls.convert_output_format_cartesian_to_polar(pred_dict)
                    pred_labels = self._feat_cls.segment_labels(pred_dict, self._ref_labels[pred_file][1])

                    # Calculated scores
                    eval.update_seld_scores(pred_labels, self._ref_labels[pred_file][0])

                # Overall SED and DOA scores
                ER, F, LE, LR, seld_scr, classwise_results = eval.compute_seld_scores()

                print('\nAverage score for {} {} data using {} coordinates'.format(score_type, 'fold' if score_type=='all' else split_key, 'Polar' if self._use_polar_format else 'Cartesian' ))
                print('SELD score (early stopping metric): {:0.2f}'.format(seld_scr))
                print('SED metrics: Error rate: {:0.2f}, F-score:{:0.1f}'.format(ER, 100*F))
                print('DOA metrics: Localization error: {:0.1f}, Localization Recall: {:0.1f}'.format(LE, 100*LR))

def reshape_3Dto2D(A):
    return A.reshape(A.shape[0] * A.shape[1], A.shape[2])


if __name__ == "__main__":
    pred_output_format_files = 'results/1_foa_dev_20220316104401_test' # Path of the DCASEoutput format files
    params = parameters.get_params()
    # Compute just the DCASE final results 
    score_obj = ComputeSELDResults(params)
    ER, F, LE, LR, seld_scr, classwise_test_scr = score_obj.get_SELD_Results(pred_output_format_files)
    print('SELD score (early stopping metric): {:0.2f}'.format(seld_scr))
    print('SED metrics: Error rate: {:0.2f}, F-score:{:0.1f}'.format(ER, 100*F))
    print('DOA metrics: Localization error: {:0.1f}, Localization Recall: {:0.1f}'.format(LE, 100*LR))
    if params['average']=='macro':
        print('Classwise results on unseen test data')
        print('Class\tER\tF\tLE\tLR\tSELD_score')
        for cls_cnt in range(params['unique_classes']):
            print('{}\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.2f}'.format(cls_cnt, classwise_test_scr[0][cls_cnt], classwise_test_scr[1][cls_cnt], classwise_test_scr[2][cls_cnt], classwise_test_scr[3][cls_cnt], classwise_test_scr[4][cls_cnt]))


    # UNCOMMENT to Compute DCASE results along with room-wise performance
    # score_obj.get_consolidated_SELD_results(pred_output_format_files)

