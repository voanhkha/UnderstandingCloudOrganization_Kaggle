from sklearn.metrics import precision_recall_curve, auc
from tqdm import tqdm
from classifier.model import get_model
from sklearn.model_selection import StratifiedKFold
from classifier.preprocess import preprocess
from classifier.generator import DataGenenerator
import os
import pandas as pd
import numpy as np
import argparse
import sys



def get_threshold_for_recall(y_true, y_pred, class_i, recall_threshold=0.85, precision_threshold=0.90):
    precision, recall, thresholds = precision_recall_curve(y_true[:, class_i], y_pred[:, class_i])
    pr_auc = auc(recall, precision)
    i = len(thresholds) - 1
    best_recall_threshold = None
    while best_recall_threshold is None:
        next_threshold = thresholds[i]
        next_recall = recall[i]
        if next_recall >= recall_threshold:
            best_recall_threshold = next_threshold
        i -= 1

    # consice, even though unnecessary passing through all the values
    best_precision_threshold = [thres for prec, thres in zip(precision, thresholds) if prec >= precision_threshold][0]

    return best_recall_threshold, best_precision_threshold, pr_auc


def threshold_search(cls_model='b2', shape=(320,320)):
    max_fold = 3
    model = get_model(cls_model, shape=shape)
    kfold = StratifiedKFold(n_splits=4, random_state=133, shuffle=True)
    train_df, img_2_vector = preprocess()
    oof_true = []
    oof_pred = []

    for n_fold, (train_indices, val_indices) in enumerate(
            kfold.split(train_df['Image'].values, train_df['Class'].map(lambda x: str(sorted(list(x)))))):
        val_imgs = train_df['Image'].values[val_indices]

        if n_fold <= max_fold:
            data_generator_val = DataGenenerator(val_imgs, shuffle=False,
                                                 resized_height=shape[0], resized_width=shape[1],
                                                 img_2_ohe_vector=img_2_vector)

            model.load_weights('classifier/checkpoints/' + cls_model + '_' + str(n_fold) + '.h5')

            y_pred = model.predict_generator(data_generator_val, workers=12, verbose=1)
            y_true = data_generator_val.get_labels()

            oof_true.extend(y_true)
            oof_pred.extend(y_pred)

    oof_true = np.asarray(oof_true)
    oof_pred = np.asarray(oof_pred)
    print(oof_true.shape)
    print(oof_pred.shape)
    recall_thresholds = dict()
    precision_thresholds = dict()
    threshold_values = np.arange(0,1,0.01)
    for i, class_name in tqdm(enumerate(class_names)):
        recall_thresholds[class_name], precision_thresholds[class_name], auc = get_threshold_for_recall(oof_true, oof_pred, i)
        # best_auc = 0
        # for t in threshold_values:
        #      r , p , auc = get_threshold_for_recall(oof_true, oof_pred, i, recall_threshold=t)
        #      if auc >= best_auc:
        #         recall_thresholds[class_name], precision_thresholds[class_name] = r,p
        #         best_auc = auc

        print('Best auc {} for class {}'.format(auc,class_name))

    return recall_thresholds


def postprocess_submission(cls_model='b2', shape=(320,320), submission_file=None):
    recall_thresholds = threshold_search(cls_model, shape)
    print(recall_thresholds)
    model = get_model(cls_model, shape=shape)
    data_generator_test = DataGenenerator(folder_imgs='../../dados/test_images', shuffle=False, batch_size=1,
                                          resized_height=shape[0], resized_width=shape[1])

    for i in range(4):
        model.load_weights('classifier/checkpoints/' + cls_model + '_' + str(i) + '.h5')
        if i == 0:
            y_pred_test = model.predict_generator(data_generator_test, workers=12, verbose=1)
        else:
            y_pred_test += model.predict_generator(data_generator_test, workers=12, verbose=1)

    y_pred_test /= 4

    image_labels_empty = set()
    for i, (img, predictions) in enumerate(zip(os.listdir('../../dados/test_images'), y_pred_test)):
        for class_i, class_name in enumerate(class_names):
            if predictions[class_i] < recall_thresholds[class_name]:
                image_labels_empty.add(f'{img}_{class_name}')

    submission = pd.read_csv(submission_file)

    predictions_nonempty = set(submission.loc[~submission['EncodedPixels'].isnull(), 'Image_Label'].values)
    print(f'{len(image_labels_empty.intersection(predictions_nonempty))} masks would be removed')
    submission.loc[submission['Image_Label'].isin(image_labels_empty), 'EncodedPixels'] = np.nan
    submission.to_csv('../submissions/submission_segmentation_and_classifier.csv', index=None)

def parse_args(args):
    """ Parse the arguments.
    """
    parser = argparse.ArgumentParser(description='Predict script')


    parser.add_argument('--model', help='Classification model', default='b2')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--shape', help='Shape of resized images', default=(256,256), type=tuple)
    parser.add_argument("--file", default=None, type=str)
    parser.add_argument("--cpu", default=False, type=bool)

    return parser.parse_args(args)

if __name__ == '__main__':
    args = sys.argv[1:]
    args = parse_args(args)
    class_names = ['Fish', 'Flower', 'Sugar', 'Gravel']

    if args.cpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = ""


    postprocess_submission(args.model,args.shape, args.file)