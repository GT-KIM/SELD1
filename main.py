import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import numpy as np
import matplotlib.pyplot as plot
import time
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from time import gmtime, strftime

import scripts.cls_feature_class as cls_feature_class
import scripts.cls_data_generator as cls_data_generator

import scripts.seldnet_model as seldnet_model
import scripts.KU_Han_model as KU_Han_model
import proposed_model.Proposed_model2 as Proposed_model2
import config.parameters as parameters
from scripts.cls_compute_seld_results import ComputeSELDResults, reshape_3Dto2D
from scripts.SELD_evaluation_metrics import distance_between_cartesian_coordinates

from scripts.specmix import *


def parse_config(config_path):
    class ConfigClass:
        def __init__(self, **entries):
            for key, value in entries.items():
                if isinstance(value, dict):
                    value = ConfigClass(**value)
                self.__dict__.update({key: value})

    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = ConfigClass(**config)

        return config


def get_accdoa_labels(accdoa_in, nb_classes):
    x, y, z = accdoa_in[:, :, :nb_classes], accdoa_in[:, :, nb_classes:2 * nb_classes], accdoa_in[:, :, 2 * nb_classes:]
    sed = np.sqrt(x ** 2 + y ** 2 + z ** 2) > 0.5

    return sed, accdoa_in


def get_multi_accdoa_labels(accdoa_in, nb_classes):
    """
    Args:
        accdoa_in:  [batch_size, frames, num_track*num_axis*num_class=3*3*12]
        nb_classes: scalar
    Return:
        sedX:       [batch_size, frames, num_class=12]
        doaX:       [batch_size, frames, num_axis*num_class=3*12]
    """
    x0, y0, z0 = accdoa_in[:, :, :1 * nb_classes], accdoa_in[:, :, 1 * nb_classes:2 * nb_classes], accdoa_in[:, :,
                                                                                                   2 * nb_classes:3 * nb_classes]
    sed0 = np.sqrt(x0 ** 2 + y0 ** 2 + z0 ** 2) > 0.5
    doa0 = accdoa_in[:, :, :3 * nb_classes]

    x1, y1, z1 = accdoa_in[:, :, 3 * nb_classes:4 * nb_classes], accdoa_in[:, :,
                                                                 4 * nb_classes:5 * nb_classes], accdoa_in[:, :,
                                                                                                 5 * nb_classes:6 * nb_classes]
    sed1 = np.sqrt(x1 ** 2 + y1 ** 2 + z1 ** 2) > 0.5
    doa1 = accdoa_in[:, :, 3 * nb_classes: 6 * nb_classes]

    x2, y2, z2 = accdoa_in[:, :, 6 * nb_classes:7 * nb_classes], accdoa_in[:, :,
                                                                 7 * nb_classes:8 * nb_classes], accdoa_in[:, :,
                                                                                                 8 * nb_classes:]
    sed2 = np.sqrt(x2 ** 2 + y2 ** 2 + z2 ** 2) > 0.5
    doa2 = accdoa_in[:, :, 6 * nb_classes:]

    return sed0, doa0, sed1, doa1, sed2, doa2


def determine_similar_location(sed_pred0, sed_pred1, doa_pred0, doa_pred1, class_cnt, thresh_unify, nb_classes):
    if (sed_pred0 == 1) and (sed_pred1 == 1):
        if distance_between_cartesian_coordinates(doa_pred0[class_cnt], doa_pred0[class_cnt + 1 * nb_classes],
                                                  doa_pred0[class_cnt + 2 * nb_classes],
                                                  doa_pred1[class_cnt], doa_pred1[class_cnt + 1 * nb_classes],
                                                  doa_pred1[class_cnt + 2 * nb_classes]) < thresh_unify:
            return 1
        else:
            return 0
    else:
        return 0


def test_epoch(data_generator, model, criterion, dcase_output_folder, params, device):
    # Number of frames for a 60 second audio with 100ms hop length = 600 frames
    # Number of frames in one batch (batch_size* sequence_length) consists of all the 600 frames above with zero padding in the remaining frames
    test_filelist = data_generator.get_filelist()

    nb_test_batches, test_loss = 0, 0.
    model.eval()
    file_cnt = 0
    with torch.no_grad():
        for data, target in data_generator.generate():
            # load one batch of data
            data, target = torch.tensor(data).to(device).float(), torch.tensor(target).to(device).float()

            # process the batch of data based on chosen mode
            output = model(data)
            loss = criterion(output, target)
            if params['multi_accdoa'] is True:
                sed_pred0, doa_pred0, sed_pred1, doa_pred1, sed_pred2, doa_pred2 = get_multi_accdoa_labels(
                    output.detach().cpu().numpy(), params['unique_classes'])
                sed_pred0 = reshape_3Dto2D(sed_pred0)
                doa_pred0 = reshape_3Dto2D(doa_pred0)
                sed_pred1 = reshape_3Dto2D(sed_pred1)
                doa_pred1 = reshape_3Dto2D(doa_pred1)
                sed_pred2 = reshape_3Dto2D(sed_pred2)
                doa_pred2 = reshape_3Dto2D(doa_pred2)
            else:
                sed_pred, doa_pred = get_accdoa_labels(output.detach().cpu().numpy(), params['unique_classes'])
                sed_pred = reshape_3Dto2D(sed_pred)
                doa_pred = reshape_3Dto2D(doa_pred)

            # dump SELD results to the correspondin file
            output_file = os.path.join(dcase_output_folder, test_filelist[file_cnt].replace('.npy', '.csv'))
            file_cnt += 1
            output_dict = {}
            if params['multi_accdoa'] is True:
                for frame_cnt in range(sed_pred0.shape[0]):
                    for class_cnt in range(sed_pred0.shape[1]):
                        # determine whether track0 is similar to track1
                        flag_0sim1 = determine_similar_location(sed_pred0[frame_cnt][class_cnt],
                                                                sed_pred1[frame_cnt][class_cnt], doa_pred0[frame_cnt],
                                                                doa_pred1[frame_cnt], class_cnt, params['thresh_unify'],
                                                                params['unique_classes'])
                        flag_1sim2 = determine_similar_location(sed_pred1[frame_cnt][class_cnt],
                                                                sed_pred2[frame_cnt][class_cnt], doa_pred1[frame_cnt],
                                                                doa_pred2[frame_cnt], class_cnt, params['thresh_unify'],
                                                                params['unique_classes'])
                        flag_2sim0 = determine_similar_location(sed_pred2[frame_cnt][class_cnt],
                                                                sed_pred0[frame_cnt][class_cnt], doa_pred2[frame_cnt],
                                                                doa_pred0[frame_cnt], class_cnt, params['thresh_unify'],
                                                                params['unique_classes'])
                        # unify or not unify according to flag
                        if flag_0sim1 + flag_1sim2 + flag_2sim0 == 0:
                            if sed_pred0[frame_cnt][class_cnt] > 0.5:
                                if frame_cnt not in output_dict:
                                    output_dict[frame_cnt] = []
                                output_dict[frame_cnt].append([class_cnt, doa_pred0[frame_cnt][class_cnt],
                                                               doa_pred0[frame_cnt][
                                                                   class_cnt + params['unique_classes']],
                                                               doa_pred0[frame_cnt][
                                                                   class_cnt + 2 * params['unique_classes']]])
                            if sed_pred1[frame_cnt][class_cnt] > 0.5:
                                if frame_cnt not in output_dict:
                                    output_dict[frame_cnt] = []
                                output_dict[frame_cnt].append([class_cnt, doa_pred1[frame_cnt][class_cnt],
                                                               doa_pred1[frame_cnt][
                                                                   class_cnt + params['unique_classes']],
                                                               doa_pred1[frame_cnt][
                                                                   class_cnt + 2 * params['unique_classes']]])
                            if sed_pred2[frame_cnt][class_cnt] > 0.5:
                                if frame_cnt not in output_dict:
                                    output_dict[frame_cnt] = []
                                output_dict[frame_cnt].append([class_cnt, doa_pred2[frame_cnt][class_cnt],
                                                               doa_pred2[frame_cnt][
                                                                   class_cnt + params['unique_classes']],
                                                               doa_pred2[frame_cnt][
                                                                   class_cnt + 2 * params['unique_classes']]])
                        elif flag_0sim1 + flag_1sim2 + flag_2sim0 == 1:
                            if frame_cnt not in output_dict:
                                output_dict[frame_cnt] = []
                            if flag_0sim1:
                                if sed_pred2[frame_cnt][class_cnt] > 0.5:
                                    output_dict[frame_cnt].append([class_cnt, doa_pred2[frame_cnt][class_cnt],
                                                                   doa_pred2[frame_cnt][
                                                                       class_cnt + params['unique_classes']],
                                                                   doa_pred2[frame_cnt][
                                                                       class_cnt + 2 * params['unique_classes']]])
                                doa_pred_fc = (doa_pred0[frame_cnt] + doa_pred1[frame_cnt]) / 2
                                output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt],
                                                               doa_pred_fc[class_cnt + params['unique_classes']],
                                                               doa_pred_fc[class_cnt + 2 * params['unique_classes']]])
                            elif flag_1sim2:
                                if sed_pred0[frame_cnt][class_cnt] > 0.5:
                                    output_dict[frame_cnt].append([class_cnt, doa_pred0[frame_cnt][class_cnt],
                                                                   doa_pred0[frame_cnt][
                                                                       class_cnt + params['unique_classes']],
                                                                   doa_pred0[frame_cnt][
                                                                       class_cnt + 2 * params['unique_classes']]])
                                doa_pred_fc = (doa_pred1[frame_cnt] + doa_pred2[frame_cnt]) / 2
                                output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt],
                                                               doa_pred_fc[class_cnt + params['unique_classes']],
                                                               doa_pred_fc[class_cnt + 2 * params['unique_classes']]])
                            elif flag_2sim0:
                                if sed_pred1[frame_cnt][class_cnt] > 0.5:
                                    output_dict[frame_cnt].append([class_cnt, doa_pred1[frame_cnt][class_cnt],
                                                                   doa_pred1[frame_cnt][
                                                                       class_cnt + params['unique_classes']],
                                                                   doa_pred1[frame_cnt][
                                                                       class_cnt + 2 * params['unique_classes']]])
                                doa_pred_fc = (doa_pred2[frame_cnt] + doa_pred0[frame_cnt]) / 2
                                output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt],
                                                               doa_pred_fc[class_cnt + params['unique_classes']],
                                                               doa_pred_fc[class_cnt + 2 * params['unique_classes']]])
                        elif flag_0sim1 + flag_1sim2 + flag_2sim0 >= 2:
                            if frame_cnt not in output_dict:
                                output_dict[frame_cnt] = []
                            doa_pred_fc = (doa_pred0[frame_cnt] + doa_pred1[frame_cnt] + doa_pred2[frame_cnt]) / 3
                            output_dict[frame_cnt].append(
                                [class_cnt, doa_pred_fc[class_cnt], doa_pred_fc[class_cnt + params['unique_classes']],
                                 doa_pred_fc[class_cnt + 2 * params['unique_classes']]])
            else:
                for frame_cnt in range(sed_pred.shape[0]):
                    for class_cnt in range(sed_pred.shape[1]):
                        if sed_pred[frame_cnt][class_cnt] > 0.5:
                            if frame_cnt not in output_dict:
                                output_dict[frame_cnt] = []
                            output_dict[frame_cnt].append([class_cnt, doa_pred[frame_cnt][class_cnt],
                                                           doa_pred[frame_cnt][class_cnt + params['unique_classes']],
                                                           doa_pred[frame_cnt][
                                                               class_cnt + 2 * params['unique_classes']]])
            data_generator.write_output_format_file(output_file, output_dict)

            test_loss += loss.item()
            nb_test_batches += 1
            if params['quick_test'] and nb_test_batches > 15:
                break
            #if nb_test_batches == 8 :
            #    break
        test_loss /= nb_test_batches
    print(nb_test_batches)

    return test_loss


def train_epoch(data_generator_A, data_generator_B, optimizer, model, criterion, params, device):
    nb_train_batches, train_loss = 0, 0.
    model.train()
    for batch1, batch2 in zip(data_generator_A.generate(), data_generator_B.generate()):
    #for batch1 in data_generator_A.generate() :
        data1, target1 = batch1
        data2, target2 = batch2
        data, target = specmix(data1, data2, target1, target2)

        # load one batch of data
        data, target = torch.tensor(data).to(device).float(), torch.tensor(target).to(device).float()

        optimizer.zero_grad()

        # process the batch of data based on chosen mode
        output = model(data)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        nb_train_batches += 1

        if params['quick_test'] and nb_train_batches > 15 :
            break
        if nb_train_batches == 2000 :
            break

    train_loss /= nb_train_batches
    print(nb_train_batches)

    return train_loss


def main():
    args = parse_config('./config/config.yaml')
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.autograd.set_detect_anomaly(True)

    params = parameters.get_params()

    # Training setup
    test_splits = [[4]]
    val_splits = [[4]]
    train_splits = [[1, 2, 3]]

    for split_cnt, split in enumerate(test_splits):
        print('\n\n---------------------------------------------------------------------------------------------------')
        print(
            '------------------------------------      SPLIT {}   -----------------------------------------------'.format(
                split))
        print('---------------------------------------------------------------------------------------------------')

        # Unique name for the run
        loc_feat = params['dataset']
        loc_output = 'multiaccdoa' if params['multi_accdoa'] else 'accdoa'

        cls_feature_class.create_folder(params['model_dir'])
        unique_name = '{}_split{}_{}_{}'.format(
            params['mode'], split_cnt, loc_output, loc_feat
        )
        model_name = '{}_model.h5'.format(os.path.join(params['model_dir'], unique_name))
        print("unique_name: {}\n".format(unique_name))

        # Load train and validation data
        print('Loading training dataset:')
        data_gen_train_A = cls_data_generator.DataGenerator(
            params=params, split=train_splits[split_cnt]
        )
        data_gen_train_B = cls_data_generator.DataGenerator(params=params, split=train_splits[split_cnt])

        print('Loading validation dataset:')
        data_gen_val = cls_data_generator.DataGenerator(
            params=params, split=val_splits[split_cnt], shuffle=False, per_file=True, is_test=True
        )

        # Collect i/o data size and load model configuration
        data_in, data_out = data_gen_train_A.get_data_sizes()
        #model = seldnet_model.CRNN(data_in, data_out, params).to(device)
        #model = KU_Han_model.Network(data_out).to(device)
        model = Proposed_model2.Networks(data_out).to(device)
        if params['finetune_mode']:
            print('Running in finetuning mode. Initializing the model to the weights - {}'.format(
                params['pretrained_model_weights']))
            model.load_state_dict(torch.load(params['pretrained_model_weights'], map_location='cpu'))

        if params['continue_train'] :
            print('continue train')
            model.load_state_dict(torch.load(model_name, map_location=device))

        print('---------------- SELD-net -------------------')
        print('FEATURES:\n\tdata_in: {}\n\tdata_out: {}\n'.format(data_in, data_out))
        print(
            'MODEL:\n\tdropout_rate: {}\n\tCNN: nb_cnn_filt: {}, f_pool_size{}, t_pool_size{}\n\trnn_size: {}, fnn_size: {}\n'.format(
                params['dropout_rate'], params['nb_cnn2d_filt'], params['f_pool_size'], params['t_pool_size'],
                params['rnn_size'],
                params['fnn_size']))
        print(model)

        # Dump results in DCASE output format for calculating final scores
        dcase_output_val_folder = os.path.join(params['dcase_output_dir'],
                                               '{}_{}_val'.format(unique_name, strftime("%Y%m%d%H%M%S", gmtime())))
        cls_feature_class.delete_and_create_folder(dcase_output_val_folder)
        print('Dumping recording-wise val results in: {}'.format(dcase_output_val_folder))

        # Initialize evaluation metric class
        score_obj = ComputeSELDResults(params)

        # start training
        best_val_epoch = -1
        best_ER, best_F, best_LE, best_LR, best_seld_scr = 1., 0., 180., 0., 9999
        patience_cnt = 0

        nb_epoch = 100 if params['quick_test'] else params['nb_epochs']
        optimizer = optim.Adam(model.parameters(), lr=params['lr'])
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [0, 1, 2, 3], gamma=0.5)

        if params['multi_accdoa'] is True:
            criterion = seldnet_model.MSELoss_ADPIT()
        else:
            criterion = nn.MSELoss()

        recent_train_loss = [99999, 99999, 99999, 99999, 99999]
        for epoch_cnt in range(nb_epoch):
            # ---------------------------------------------------------------------
            # TRAINING
            # ---------------------------------------------------------------------
            start_time = time.time()
            train_loss = train_epoch(data_gen_train_A, data_gen_train_B, optimizer, model, criterion, params, device)
            train_time = time.time() - start_time
            avg_train_loss = np.mean(recent_train_loss)
            if train_loss > avg_train_loss :
                lr_scheduler.step()
                recent_train_loss = [99999, 99999, 99999, 99999, 99999]
            recent_train_loss.pop()
            recent_train_loss.append(train_loss)
            if epoch_cnt % 1 == 0 :
                # ---------------------------------------------------------------------
                # VALIDATION
                # ---------------------------------------------------------------------
                start_time = time.time()
                val_loss = test_epoch(data_gen_val, model, criterion, dcase_output_val_folder, params, device)

                # Calculate the DCASE 2021 metrics - Location-aware detection and Class-aware localization scores
                val_ER, val_F, val_LE, val_LR, val_seld_scr, classwise_val_scr = score_obj.get_SELD_Results_Augmented(
                    dcase_output_val_folder)

                val_time = time.time() - start_time
                model_name_epoch = '{}_model_{}.h5'.format(os.path.join(params['model_dir'], unique_name), epoch_cnt)
                torch.save(model.state_dict(), model_name_epoch)

                # Save model if loss is good
                if val_seld_scr <= best_seld_scr:
                    best_val_epoch, best_ER, best_F, best_LE, best_LR, best_seld_scr = epoch_cnt, val_ER, val_F, val_LE, val_LR, val_seld_scr
                    torch.save(model.state_dict(), model_name)

                # Print stats
                print(
                    'epoch: {}, time: {:0.2f}/{:0.2f}, '
                    # 'train_loss: {:0.2f}, val_loss: {:0.2f}, '
                    'train_loss: {:0.4f}, val_loss: {:0.4f}, '
                    'ER/F/LE/LR/SELD: {}, '
                    'best_val_epoch: {} {}'.format(
                        epoch_cnt, train_time, val_time,
                        train_loss, val_loss,
                        '{:0.2f}/{:0.2f}/{:0.2f}/{:0.2f}/{:0.2f}'.format(val_ER, val_F, val_LE, val_LR, val_seld_scr),
                        best_val_epoch,
                        '({:0.2f}/{:0.2f}/{:0.2f}/{:0.2f}/{:0.2f})'.format(best_ER, best_F, best_LE, best_LR,
                                                                           best_seld_scr))
                )

            patience_cnt += 1
            if patience_cnt > params['patience']:
                break

        # ---------------------------------------------------------------------
        # Evaluate on unseen test data
        # ---------------------------------------------------------------------
        print('Load best model weights')
        model.load_state_dict(torch.load(model_name, map_location='cpu'))

        print('Loading unseen test dataset:')
        data_gen_test = cls_data_generator.DataGenerator(
            params=params, split=test_splits[split_cnt], shuffle=False, per_file=True, is_test=True
        )

        # Dump results in DCASE output format for calculating final scores
        dcase_output_test_folder = os.path.join(params['dcase_output_dir'],
                                                '{}_{}_test'.format(unique_name, strftime("%Y%m%d%H%M%S", gmtime())))
        cls_feature_class.delete_and_create_folder(dcase_output_test_folder)

        if not os.path.isdir(os.path.join(params['dcase_output_dir'],'results')) :
            os.makedirs(os.path.join(params['dcase_output_dir'],'results'))
        f = open(os.path.join(params['dcase_output_dir'],'results', '{}_{}_test.txt'.format(unique_name, strftime("%Y%m%d%H%M%S", gmtime()))), 'w')
        print('Dumping recording-wise test results in: {}'.format(dcase_output_test_folder))
        f.write('Dumping recording-wise test results in: {}\n'.format(dcase_output_test_folder))


        test_loss = test_epoch(data_gen_test, model, criterion, dcase_output_test_folder, params, device)

        test_ER, test_F, test_LE, test_LR, test_seld_scr, classwise_test_scr = score_obj.get_SELD_Results_Augmented_Full(
            dcase_output_test_folder)

        print(
            'test_loss: {:0.2f}, ER/F/LE/LR/SELD: {}\n'.format(
                test_loss,
                '{:0.2f}/{:0.2f}/{:0.2f}/{:0.2f}/{:0.2f}'.format(test_ER, test_F, test_LE, test_LR, test_seld_scr))
        )
        f.write(
            'test_loss: {:0.2f}, ER/F/LE/LR/SELD: {}\n'.format(
                test_loss,
                '{:0.2f}/{:0.2f}/{:0.2f}/{:0.2f}/{:0.2f}'.format(test_ER, test_F, test_LE, test_LR, test_seld_scr))
        )

        if params['average'] == 'macro':
            print('Classwise results on unseen test data')
            print('Class\tER\tF\tLE\tLR\tSELD_score')
            for cls_cnt in range(params['unique_classes']):
                print('{}\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.2f}'.format(cls_cnt, classwise_test_scr[0][cls_cnt],
                                                                               classwise_test_scr[1][cls_cnt],
                                                                               classwise_test_scr[2][cls_cnt],
                                                                               classwise_test_scr[3][cls_cnt],
                                                                               classwise_test_scr[4][cls_cnt]))
            f.write('Classwise results on unseen test data\n')
            f.write('Class\tER\tF\tLE\tLR\tSELD_score\n')
            for cls_cnt in range(params['unique_classes']):
                f.write('{}\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.2f}\n'.format(cls_cnt, classwise_test_scr[0][cls_cnt],
                                                                               classwise_test_scr[1][cls_cnt],
                                                                               classwise_test_scr[2][cls_cnt],
                                                                               classwise_test_scr[3][cls_cnt],
                                                                               classwise_test_scr[4][cls_cnt]))
        f.close()


if __name__ == "__main__":
    main()

