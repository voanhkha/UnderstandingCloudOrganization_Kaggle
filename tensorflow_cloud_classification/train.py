from keras_radam import RAdam
from sklearn.model_selection import StratifiedKFold
from albumentations import Compose, VerticalFlip, HorizontalFlip, Rotate, GridDistortion
import sys
import argparse
import os
import pandas as pd
from keras.utils import Sequence
from copy import deepcopy
import random
import numpy as np
import cv2
import efficientnet.keras as efn
from keras.layers import Dense
from keras.models import Model
import keras.backend as K
import glob
from keras.callbacks import Callback
import keras
from sklearn.metrics import precision_recall_curve, auc
import multiprocessing

#num_cores = multiprocessing.cpu_count()

class PrAucCallback(Callback):
    def __init__(self, data_generator, num_workers=1,
                 early_stopping_patience=5,
                 plateau_patience=3, reduction_rate=0.5,
                 stage='train', checkpoints_path='classifier/checkpoints/',
                 checkpoint_name=''):
        super(Callback, self).__init__()
        self.data_generator = data_generator
        self.num_workers = num_workers
        self.class_names = ['Fish', 'Flower', 'Sugar', 'Gravel']
        self.history = [[] for _ in range(len(self.class_names) + 1)] # to store per each class and also mean PR AUC
        self.early_stopping_patience = early_stopping_patience
        self.plateau_patience = plateau_patience
        self.reduction_rate = reduction_rate
        self.stage = stage
        self.best_pr_auc = -float('inf')
        if not os.path.exists(checkpoints_path):
            os.makedirs(checkpoints_path)
        self.checkpoints_path = checkpoints_path
        self.checkpoint_name = checkpoint_name

    def compute_pr_auc(self, y_true, y_pred):
        pr_auc_mean = 0
        print(f"\n{'# ' *30}\n")
        for class_i in range(len(self.class_names)):
            precision, recall, _ = precision_recall_curve(y_true[:, class_i], y_pred[:, class_i])
            pr_auc = auc(recall, precision)
            pr_auc_mean += pr_auc /len(self.class_names)
            print(f"PR AUC {self.class_names[class_i]}, {self.stage}: {pr_auc:.3f}\n")
            self.history[class_i].append(pr_auc)
        print(f"\n{'# ' *20}\n PR AUC mean, {self.stage}: {pr_auc_mean:.3f}\n{'# ' *20}\n")
        self.history[-1].append(pr_auc_mean)
        return pr_auc_mean

    def is_patience_lost(self, patience):
        if len(self.history[-1]) > patience:
            best_performance = max(self.history[-1][-(patience + 1):-1])
            return best_performance == self.history[-1][-(patience + 1)] and best_performance >= self.history[-1][-1]

    def early_stopping_check(self, pr_auc_mean):
        if self.is_patience_lost(self.early_stopping_patience):
            self.model.stop_training = True

    def model_checkpoint(self, pr_auc_mean, epoch):
        if pr_auc_mean > self.best_pr_auc:
            # remove previous checkpoints to save space
            for checkpoint in glob.glob(os.path.join(self.checkpoints_path, 'classifier_densenet169_epoch_*')):
                os.remove(checkpoint)
            self.best_pr_auc = pr_auc_mean
            self.model.save(os.path.join(self.checkpoints_path, self.checkpoint_name + '.h5'))
            print(f"\n{'# ' *20}\nSaved new checkpoint\n{'# ' *20}\n")

    def reduce_lr_on_plateau(self):
        if self.is_patience_lost(self.plateau_patience):
            new_lr = float(keras.backend.get_value(self.model.optimizer.lr)) * self.reduction_rate
            keras.backend.set_value(self.model.optimizer.lr, new_lr)
            print(f"\n{'# ' *20}\nReduced learning rate to {new_lr}.\n{'# ' *20}\n")

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict_generator(self.data_generator, workers=self.num_workers)
        y_true = self.data_generator.get_labels()
        # estimate AUC under precision recall curve for each class
        pr_auc_mean = self.compute_pr_auc(y_true, y_pred)

        if self.stage == 'val':
            # early stop after early_stopping_patience=4 epochs of no improvement in mean PR AUC
            self.early_stopping_check(pr_auc_mean)

            # save a model with the best PR AUC in validation
            self.model_checkpoint(pr_auc_mean, epoch)

            # reduce learning rate on PR AUC plateau
            self.reduce_lr_on_plateau()

    def get_pr_auc_history(self):
        return self.history

def get_model(model='b2', shape=(320,320)):
    K.clear_session()
    h,w = shape
    if model == 'b0':
        base_model = efn.EfficientNetB0(weights='imagenet', include_top=False, pooling='avg', input_shape=(h, w, 3))

    elif model == 'b1':
        base_model = efn.EfficientNetB1(weights='imagenet', include_top=False, pooling='avg', input_shape=(h, w, 3))

    elif model == 'b2':
        base_model = efn.EfficientNetB2(weights='imagenet', include_top=False, pooling='avg', input_shape=(h, w, 3))

    elif model == 'b3':
        base_model =  efn.EfficientNetB3(weights='imagenet', include_top=False, pooling='avg', input_shape=(h, w, 3))

    elif model == 'b4':
        base_model =  efn.EfficientNetB4(weights='imagenet', include_top=False, pooling='avg', input_shape=(h, w, 3))

    elif model == 'b5':
        base_model =  efn.EfficientNetB5(weights='imagenet', include_top=False, pooling='avg', input_shape=(h, w, 3))

    elif model == 'b6':
        base_model =  efn.EfficientNetB6(weights='imagenet', include_top=False, pooling='avg', input_shape=(h, w, 3))

    else:
        base_model =  efn.EfficientNetB7(weights='imagenet', include_top=False, pooling='avg', input_shape=(h, w, 3))

    x = base_model.output
    y_pred = Dense(4, activation='sigmoid')(x)
    return Model(inputs=base_model.input, outputs=y_pred)

class DataGenenerator(Sequence):
    def __init__(self, images_list=None, folder_imgs='/data/khavo/cloud_kaggle/input/train_images', #'D:/cloud_kaggle_data/input/train_images',
                 img_2_ohe_vector=None,batch_size=32, shuffle=True, augmentation=None,
                 resized_height=260, resized_width=260, num_channels=3):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augmentation = augmentation
        if images_list is None:
            self.images_list = os.listdir(folder_imgs)
        else:
            self.images_list = deepcopy(images_list)
        self.folder_imgs = folder_imgs
        self.len = len(self.images_list) // self.batch_size
        self.resized_height = resized_height
        self.resized_width = resized_width
        self.num_channels = num_channels
        self.num_classes = 4
        self.is_test = not 'train' in folder_imgs

        self.img_2_ohe_vector = img_2_ohe_vector
        if not shuffle and not self.is_test:
            self.labels = [self.img_2_ohe_vector[img] for img in self.images_list[:self.len * self.batch_size]]

    def __len__(self):
        return self.len

    def on_epoch_start(self):
        if self.shuffle:
            random.shuffle(self.images_list)

    def __getitem__(self, idx):
        current_batch = self.images_list[idx * self.batch_size: (idx + 1) * self.batch_size]
        X = np.empty((self.batch_size, self.resized_height, self.resized_width, self.num_channels))
        y = np.empty((self.batch_size, self.num_classes))

        for i, image_name in enumerate(current_batch):
            path = os.path.join(self.folder_imgs, image_name)
            img = cv2.resize(cv2.imread(path), (self.resized_height, self.resized_width)).astype(np.float32)
            if not self.augmentation is None:
                augmented = self.augmentation(image=img)
                img = augmented['image']
            X[i, :, :, :] = img / 255.0
            if not self.is_test:
                y[i, :] = self.img_2_ohe_vector[image_name]
        return X, y

    def get_labels(self):
        if self.shuffle:
            images_current = self.images_list[:self.len * self.batch_size]
            labels = [self.img_2_ohe_vector[img] for img in images_current]
        else:
            labels = self.labels
        return np.array(labels)
def preprocess():
    train_df = pd.read_csv('/data/khavo/cloud_kaggle/input/train.csv')
    train_df = train_df[~train_df['EncodedPixels'].isnull()]
    train_df['Image'] = train_df['Image_Label'].map(lambda x: x.split('_')[0])
    train_df['Class'] = train_df['Image_Label'].map(lambda x: x.split('_')[1])
    classes = train_df['Class'].unique()
    train_df = train_df.groupby('Image')['Class'].agg(set).reset_index()
    for class_name in classes:
        train_df[class_name] = train_df['Class'].map(lambda x: 1 if class_name in x else 0)

    img_2_ohe_vector = {img: vec for img, vec in zip(train_df['Image'], train_df.iloc[:, 2:].values)}

    return train_df, img_2_ohe_vector


def train(cls_model='b2', shape=(320,320), batch_size=5, seed=1):

    kfold = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
    train_df, img_2_vector = preprocess()

    albumentations_train = Compose([
        VerticalFlip(), HorizontalFlip(), Rotate(limit=20), GridDistortion()
    ], p=1)

    for n_fold, (train_indices, val_indices) in enumerate(kfold.split(train_df['Image'].values, train_df['Class'].map(lambda x: str(sorted(list(x)))))):
        train_imgs = train_df['Image'].values[train_indices]
        val_imgs = train_df['Image'].values[val_indices]
        data_generator_train = DataGenenerator(train_imgs, augmentation=albumentations_train,
                                               resized_height=shape[0], resized_width=shape[1],
                                               img_2_ohe_vector=img_2_vector, batch_size=batch_size)

        data_generator_train_eval = DataGenenerator(train_imgs, shuffle=False,
                                                    resized_height=shape[0], resized_width=shape[1],
                                                    img_2_ohe_vector=img_2_vector, batch_size=batch_size)

        data_generator_val = DataGenenerator(val_imgs, shuffle=False,
                                             resized_height=shape[0], resized_width=shape[1],
                                             img_2_ohe_vector=img_2_vector, batch_size=batch_size)

        model = get_model(cls_model, shape=shape)

        model.compile(optimizer=RAdam(), loss='binary_crossentropy',
                      metrics=['accuracy'])

        train_metric_callback = PrAucCallback(data_generator_train_eval)
        checkpoint_name = cls_model +'_seed' + str(seed) +'_' + str(n_fold)
        val_callback = PrAucCallback(data_generator_val, stage='val', checkpoint_name=checkpoint_name)

        model.fit_generator(generator=data_generator_train,
                            validation_data=data_generator_val,
                            epochs=40,
                            callbacks=[train_metric_callback, val_callback],
                            verbose=1
                            )

def parse_args(args):
    """ Parse the arguments.
    """
    parser = argparse.ArgumentParser(description='Predict script')
    parser.add_argument('--model', help='Classification model', default='b2')
    parser.add_argument('--batch_size', default=5, type=int)
    parser.add_argument('--shape', help='Shape of resized images', default=(256,256), type=tuple)
    parser.add_argument("--cpu", default=False, type=bool)
    parser.add_argument("--seed", default=1, type=int)
    return parser.parse_args(args)

if __name__ == '__main__':
    args = sys.argv[1:]
    args = parse_args(args)

    if args.cpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = ""


    train(args.model,args.shape, batch_size=args.batch_size, seed=args.seed)