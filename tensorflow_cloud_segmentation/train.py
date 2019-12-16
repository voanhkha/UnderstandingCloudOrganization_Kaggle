import tensorflow as tf
import argparse, multiprocessing, sys, gc, itertools, keras, cv2, ast
import pandas as pd, numpy as np, albumentations as albu
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from keras_radam import RAdam
from keras.optimizers import Adam, Nadam, SGD, Optimizer
from keras.legacy import interfaces
from keras.losses import binary_crossentropy, categorical_crossentropy
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import keras.backend as K
import segmentation_models as sm
from segmentation_models.losses import bce_jaccard_loss
from skimage.exposure import adjust_gamma

### MODEL SECTION ###

def get_model(model,backbone,opt,loss,metric,shape, nb_classes=4):
    h,w = shape
    wei, act = 'imagenet', 'softmax'
    if model == 'pspnet':model = sm.PSPNet(backbone, classes=nb_classes, input_shape=(h, w, 3), activation=act,encoder_weights=wei) 
    elif model == 'fpn': model = sm.FPN(backbone,classes=nb_classes,input_shape=(h, w, 3),activation=act,encoder_weights=wei)
    elif model == 'linknet': model = sm.Linknet(backbone, classes=nb_classes,  input_shape=(h, w, 3), activation=act, encoder_weights=wei)
    elif model == 'unet': model = sm.Unet(backbone, classes=nb_classes, input_shape=(h, w, 3), activation=act,   encoder_weights=wei)
    else: raise ValueError('Unknown model specification ' + model)
    model.compile(optimizer=opt, loss=loss, metrics=[metric])
    return model

### LOSS SECTION ###

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred = K.cast(y_pred, 'float32')
    y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')
    intersection = y_true_f * y_pred_f
    score = 2. * K.sum(intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))
    return score

def dice_coef_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score

def binary_crossentropy_smoothed(y_true, y_pred):
    loss =  tf.keras.losses.binary_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0.1)
    return loss

def dice_coef_loss_bce(y_true, y_pred, dice=.25, bce=.75):
    return binary_crossentropy(y_true, y_pred) * bce + dice_coef_loss(y_true, y_pred) * dice

def focal_loss(y_true, y_pred):
    gamma = 2.0
    alpha = 0.25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    epsilon = K.epsilon()
    # clip to prevent NaN's and Inf's
    pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
    pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)
    return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))


def combo_loss(y_true, y_pred, dice=0, bce=0.5, focal=0.5):
    return binary_crossentropy_smoothed(y_true, y_pred) * bce + \
           dice_coef_loss(y_true, y_pred) * dice + \
           focal_loss(y_true,y_pred) * focal

### OPTIMIZER SECTION ###

class AdamAccumulate(Optimizer):

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=None, decay=0., amsgrad=False, accum_iters=1, **kwargs):
        if accum_iters < 1:
            raise ValueError('accum_iters must be >= 1')
        super(AdamAccumulate, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='learning_rate')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay
        self.amsgrad = amsgrad
        self.accum_iters = K.variable(accum_iters, K.dtype(self.iterations))
        self.accum_iters_float = K.cast(self.accum_iters, K.floatx())

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]
        lr = self.lr
        completed_updates = K.cast(K.tf.floordiv(self.iterations, self.accum_iters), K.floatx())
        if self.initial_decay > 0:  lr = lr * (1. / (1. + self.decay * completed_updates))
        t = completed_updates + 1
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) / (1. - K.pow(self.beta_1, t)))
        # self.iterations incremented after processing a batch
        # batch:              1 2 3 4 5 6 7 8 9
        # self.iterations:    0 1 2 3 4 5 6 7 8
        # update_switch = 1:        x       x    (if accum_iters=4)
        update_switch = K.equal((self.iterations + 1) % self.accum_iters, 0)
        update_switch = K.cast(update_switch, K.floatx())

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        gs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]

        if self.amsgrad: vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else: vhats = [K.zeros(1) for _ in params]

        self.weights = [self.iterations] + ms + vs + vhats
        for p, g, m, v, vhat, tg in zip(params, grads, ms, vs, vhats, gs):

            sum_grad = tg + g
            avg_grad = sum_grad / self.accum_iters_float

            m_t = (self.beta_1 * m) + (1. - self.beta_1) * avg_grad
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(avg_grad)

            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                p_t = p - lr_t * m_t / (K.sqrt(vhat_t) + self.epsilon)
                self.updates.append(K.update(vhat, (1 - update_switch) * vhat + update_switch * vhat_t))
            else:  p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(K.update(m, (1 - update_switch) * m + update_switch * m_t))
            self.updates.append(K.update(v, (1 - update_switch) * v + update_switch * v_t))
            self.updates.append(K.update(tg, (1 - update_switch) * sum_grad))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:  new_p = p.constraint(new_p)
            self.updates.append(K.update(p, (1 - update_switch) * p + update_switch * new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon,
                  'amsgrad': self.amsgrad}
        base_config = super(AdamAccumulate, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

### GENERATOR SECTION ###

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, train_df,
                 base_path='/home/khavo/Data/cloud_kaggle/input/train_images',
                 batch_size=32, dim=(384, 576), n_channels=3, 
                 augment=False,  random_state=2019, shuffle=True, backbone='resnet34', use_cutmix=True,
                 cloudtype=5):
        self.dim = dim
        self.batch_size = batch_size
        self.train_df = train_df
        self.nb_samples = len(train_df)
        self.base_path = base_path
        self.n_channels = n_channels
        self.augment = augment
        self.n_classes = 1 if cloudtype in [0,1,2,3] else 4
        self.shuffle = shuffle
        self.random_state = random_state
        self.preprocess_input = sm.get_preprocessing(backbone)
        self.on_epoch_end()
        self.use_cutmix = use_cutmix
        self.cloudtype = cloudtype
        np.random.seed(self.random_state)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.train_df) / self.batch_size))


    def __getitem__(self, index):

        indexes = self.indexes[index*self.batch_size : (index+1)*self.batch_size]
        X, y = self.__generate_kha(indexes)

        if self.use_cutmix:
            do_cutmix = np.random.choice([0, 1], 1, p=[0.5, 0.5])
            if do_cutmix == 1:
                indexes_augment = np.random.choice(self.nb_samples-1, len(y)) # random pick len(y) indexes from 0 to len(listIDs)
                Xc, yc = self.__generate_kha(indexes_augment) # randomly pick images for cutmix

                w, h = self.dim
                #cutmix_ratio = np.choice([1/4, 1/3, 1/2])
                w_cutmix, h_cutmix = int(w/2), int(h/2)
                w_start, h_start = int(np.random.choice(w - w_cutmix - 1 , 1)), int(np.random.choice(h - h_cutmix - 1 , 1))
                w_start2, h_start2 = int(np.random.choice(w - w_cutmix - 1 , 1)), int(np.random.choice(h - h_cutmix - 1 , 1))

                X[:, w_start:w_start+w_cutmix, h_start:h_start+h_cutmix, :] = Xc[:, w_start2:w_start2+w_cutmix, h_start2:h_start2+h_cutmix, :]
                y[:, w_start:w_start+w_cutmix, h_start:h_start+h_cutmix] = yc[:, w_start2:w_start2+w_cutmix, h_start2:h_start2+h_cutmix]

        if self.augment:  X, y = self.__augment_batch(X, y)
        return self.preprocess_input(X), y


    def on_epoch_end(self):
        self.indexes = np.arange(self.nb_samples)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __generate_kha(self, indices):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim, self.n_classes), dtype=int)

        for i, ID in enumerate(indices):
            
            im_name = self.train_df['im_id'].iloc[ID]
            #print(im_name)
            img_path = f"{self.base_path}/{im_name}"
            img = self.__load_rgb(img_path)

            rles = self.train_df['rles'].iloc[ID]
            if self.cloudtype in [0,1,2,3]: rles = [rles[self.cloudtype]]
            masks = build_masks(rles, input_shape=self.dim)

            X[i,] = img
            y[i,] = masks

        return X, y

    def __load_grayscale(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        # img = img.astype(np.float32) / 255.
        img = np.expand_dims(img, axis=-1)
        return img

    def __load_rgb(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __random_transform(self, img, masks):
        composition = albu.Compose([
                        # albu.OneOf([albu.RandomSizedCrop(min_max_height=(self.reshape[0]//2, self.reshape[0]),
                        #                                  height=self.reshape[0], width=self.reshape[1], w2h_ratio=1.5,
                        #                                  p=0.5),
                        #       albu.PadIfNeeded(min_height=self.reshape[0], min_width=self.reshape[1], p=0.5)], p=0.3),
                        albu.RandomSizedCrop(min_max_height=(self.dim[0] // 2, self.dim[0]),
                                                                   height=self.dim[0], width=self.dim[1], w2h_ratio=1.5,
                                                                   p=0.2),
                        albu.HorizontalFlip(p=.3),
                        albu.VerticalFlip(p=.3),
                        albu.ShiftScaleRotate(rotate_limit=45, shift_limit=0.15, scale_limit=0.15, p=0.1),
                        albu.OneOf([
                            albu.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                            albu.GridDistortion(p=0.5),
                            albu.OpticalDistortion(p=0.5, distort_limit=2, shift_limit=0.5)
                            ], p=0.15),
                        albu.OneOf([
                            albu.RandomContrast(),
                            albu.RandomGamma(),
                            albu.RandomBrightness(),
                            albu.Solarize()
                        ], p=0.15)
        ], p=1)

        composed = composition(image=img.astype('uint8'), mask=masks)
        aug_img = composed['image']
        aug_masks = composed['mask']
        return aug_img, aug_masks

    def __augment_batch(self, img_batch, masks_batch):
        for i in range(img_batch.shape[0]):
            img_batch[i,], masks_batch[i,] = self.__random_transform(img_batch[i,], masks_batch[i,])
        return img_batch, masks_batch

### UTIL SECTION ###

def np_resize(img, input_shape):
    height, width = input_shape
    return cv2.resize(img, (width, height))


def mask2rle(img):
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle2mask(rle, input_shape):
    width, height = input_shape[:2]
    mask = np.zeros(width * height).astype(np.uint8)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]
    for index, start in enumerate(starts): mask[int(start):int(start + lengths[index])] = 1
    return mask.reshape(height, width).T


def build_masks(rles, input_shape):
    h, w = input_shape
    depth = len(rles)
    masks = np.zeros((h, w, depth))
    for i, rle in enumerate(rles):
        if rle is not np.nan and len(rle)>0:
            masks[:, :, i] = rle2mask(rle, input_shape)
    return masks


def rle_decode(mask_rle='', shape=(1400, 2100)):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends): img[lo:hi] = 1
    return img.reshape(shape, order='F')


def make_mask(df, image_label, shape=(1400, 2100), cv_shape=(525, 350)):
    df = df.set_index('Image_Label')
    encoded_mask = df.loc[image_label, 'EncodedPixels']
    mask = np.zeros((shape[0], shape[1]), dtype=np.float32)
    if encoded_mask is not np.nan:
        mask = rle_decode(encoded_mask, shape=shape)  # original size

    return cv2.resize(mask, cv_shape)


### TRAIN SECTION ###

def train(modelname, backbone, batch_size, shape=(384,576), pseudo=None, nb_pseudo=1000, input_path='', save_path='', \
    folds_to_train=[0, 1], splits=5, seed=0, use_cutmix=True, cloudtype=5, loss='dice_bce'):

    train_df = pd.read_csv(input_path+'train_384x576.csv')
    train_df['rles'] = train_df['rles'].apply(ast.literal_eval)

    if pseudo is not None:  
        pseudo_df = pd.read_csv(input_path + pseudo + '.csv').tail(nb_pseudo)
        pseudo_df['rles'] = pseudo_df['rles'].apply(ast.literal_eval) # convert column 'rles' from type str to type list
        # because when to_csv, the column with list type is automatically converted to string type
        # i.e., [1,2,3] to '[1,2,3]'
        
    skf = StratifiedKFold(n_splits=splits, random_state=seed, shuffle=True)
    stratify_col = 'hasMask_eachtype' if cloudtype not in [0,1,2,3] else 'hasMask_type'+str(cloudtype)

    for n_fold, (train_indices, val_indices) in enumerate(skf.split(train_df, train_df[stratify_col])):
            if n_fold not in folds_to_train: continue

            print('Training fold number ',str(n_fold))

            df_train_fold = train_df.iloc[train_indices].reset_index(drop=True)
            df_valid_fold = train_df.iloc[val_indices].reset_index(drop=True)
            if pseudo is not None: df_train_fold = pd.concat([df_train_fold[['im_id','rles']], pseudo_df[['im_id','rles']]]).reset_index(drop=True)
            print('Training samples:', len(df_train_fold))
            print('Validation samples:', len(df_valid_fold))

            train_generator = DataGenerator(df_train_fold, base_path=input_path+'resized', use_cutmix=use_cutmix,
                batch_size=batch_size, augment=True, backbone=backbone, cloudtype=cloudtype)

            val_generator = DataGenerator(df_valid_fold, base_path=input_path+'resized',use_cutmix=False,
                batch_size=batch_size, augment=False, backbone=backbone, cloudtype=cloudtype)

            # opt = RAdam(lr=0.0001)
            opt = Nadam(lr=0.001)
            #opt = AdamAccumulate(lr=0.001, accum_iters=8)

            if loss == 'dice_bce': loss_fn = dice_coef_loss_bce
            if loss == 'combo': loss_fn = combo_loss

            nb_classes = 4 if cloudtype not in [0,1,2,3] else 1
            model = get_model(modelname, backbone, opt, loss_fn, dice_coef , shape, nb_classes=nb_classes)

            ckppath = save_path + modelname + '_' + backbone + '_cloudtype' + str(cloudtype) + '_bs' +str(batch_size) + '_shape' +\
                 str(shape[0])+'x'+str(shape[1]) +'_cutmix' + str(int(use_cutmix)) + '_seed' + str(seed) + '_splits' + str(splits)+ \
                     '_pseudo' + pseudo + '_nbpd' +str(nb_pseudo) + '_loss' + loss + '_fold' + str(n_fold) + '.h5'
            ckp = ModelCheckpoint(ckppath, monitor='val_dice_coef', verbose=1, save_best_only=False, mode='max',
                                         save_weights_only=True)
            es = EarlyStopping(monitor='val_dice_coef', min_delta=0.0001, patience=3, verbose=1, mode='max')
            rlr = ReduceLROnPlateau(monitor='val_dice_coef', factor=0.2, patience=2, verbose=1, mode='max', min_delta=0.0001)
            model.fit_generator(train_generator, validation_data=val_generator, callbacks=[ckp, rlr, es], epochs=50,
                #use_multiprocessing=True,
               # workers=1
            )

def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args(args):
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--model', help='Segmentation model', default='unet')
    parser.add_argument('--backbone', help='Model backbone', default='efficientnetb5', type=str)
    parser.add_argument('--batch_size', default=3, type=int)
    parser.add_argument('--shape', help='Shape of resized images', type=str)
    parser.add_argument('--use_cutmix', help='use_cutmix True of False', default=True, type=bool)
    parser.add_argument('--splits', default=5, type=int)
    parser.add_argument('--folds', help='folds to train, separated by comma', default=[0,1], type=str)
    parser.add_argument('--pseudo', help='Add extra data from test', default=None, type=str)
    parser.add_argument('--nb_pseudo', default=1000, type=int)
    parser.add_argument('--seed', help='kfold seed', default=1, type=int)
    parser.add_argument('--machine', help='cibci or home', default='home', type=str)
    parser.add_argument('--cloudtype', default=5, type=int)
    parser.add_argument('--loss',  default='dice_bce', type=str)
    return parser.parse_args(args)

if __name__ == '__main__':
    args = sys.argv[1:]
    args = parse_args(args)

    if args.machine == 'home':
        input_data_path = 'D:/cloud_kaggle_data/input/'
        save_path = 'D:/cloud_kaggle_data/trained_models/'
    elif args.machine == 'cibci':
        input_data_path = '/data/khavo/cloud_kaggle/input/'
        save_path = '/data/khavo/cloud_kaggle/trained_models/'    
    else: raise ValueError('Machine specification error')

    folds = [int(item) for item in args.folds.split(',')]
    #shape = tuple([int(item) for item in args.shape.split(',')])

    train(modelname=args.model, backbone=args.backbone, batch_size=args.batch_size, 
        pseudo=args.pseudo,nb_pseudo=args.nb_pseudo, input_path=input_data_path, 
        save_path=save_path, folds_to_train=folds, splits=args.splits, seed=args.seed, 
        use_cutmix=str2bool(args.use_cutmix), cloudtype=args.cloudtype, loss=args.loss)

    #  python train.py --model fpn --backbone efficientnetb3 --shape 384,576 --pseudo pseudo_384x576_v1
    #   --nb_pseudo 1000 --batch_size 3 --use_cutmix True
    #  --splits 4 --folds 0,1,2,3 --seed 6 --machine home --cloudtype 5