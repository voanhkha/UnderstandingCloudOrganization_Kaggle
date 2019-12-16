import os, cv2, time, numpy as np, pandas as pd
from tqdm import tqdm as tqdm
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
#import torchvision
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
import albumentations as albu
from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90, RandomCrop,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, Cutout, 
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, ElasticTransform,
    Resize, Lambda
)
#from albumentations import torch as AT
# from catalyst.data import Augmentor
# from catalyst.dl import utils
# from catalyst.data.reader import ImageReader, ScalarReader, ReaderCompose, LambdaReader
# from catalyst.dl.runner import SupervisedRunner
# from catalyst.contrib.models.segmentation import Unet
# from catalyst.dl.callbacks import DiceCallback, EarlyStoppingCallback, InferCallback, CheckpointCallback
#import segmentation_models_pytorch as smp
#from efficientunet import *
import random

RANDOM_SEED = 0
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def timer(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:0>2}".format(int(hours),int(minutes),int(seconds))

def get_best_stats(log):
    best = {'thres':[0,0,0,0], 'minsize':[0,0,0,0], 'dice':[0,0,0,0], 'dice_avg':0}
    for j, cloudtype in enumerate(['Fish', 'Flower', 'Gravel', 'Sugar']):
        cols = [c for c in log.columns if cloudtype in c]
        for c in cols: log[c] = log[c].astype('float32')
        b = log.tail(1)[cols].idxmax(axis=1).values[0]
        v = log.tail(1)[cols].max(axis=1).values[0]
        best['thres'][j] = int(b.split('_')[1])
        best['minsize'][j] = int(b.split('_')[2])
        best['dice'][j] = v
    best['dice_avg'] = np.mean(best['dice'])
    return best

class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta)
        if patience == 0: self.is_better = lambda a, b: True

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs > self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if mode == 'min':
            self.is_better = lambda a, best: a < best - min_delta
        if mode == 'max':
            self.is_better = lambda a, best: a > best + min_delta


def load_csv(path, n_splits=4, seed=0, fold_number=0, use_all_data=False):
    train = pd.read_csv(f'{path}/train.csv')
    sub = pd.read_csv(f'{path}/sample_submission.csv')

    n_train = len(os.listdir(f'{path}/train_images'))
    n_test = len(os.listdir(f'{path}/test_images'))
    print(f'There are {n_train} images in train dataset')
    print(f'There are {n_test} images in test dataset')

    train['label'] = train['Image_Label'].apply(lambda x: x.split('_')[1])
    train['im_id'] = train['Image_Label'].apply(lambda x: x.split('_')[0])
    sub['label'] = sub['Image_Label'].apply(lambda x: x.split('_')[1])
    sub['im_id'] = sub['Image_Label'].apply(lambda x: x.split('_')[0])
    # id_mask_count = train.loc[train['EncodedPixels'].isnull() == False, 'Image_Label'].apply(lambda x: x.split('_')[0]).value_counts().\
    # reset_index().rename(columns={'index': 'img_id', 'Image_Label': 'count'})

    id_mask_count = train.loc[train['EncodedPixels'].isnull() == False, 'Image_Label'].apply(lambda x: x.split('_')[0]).value_counts().\
        reset_index().rename(columns={'index': 'img_id', 'Image_Label': 'count'}).sort_values(['count', 'img_id'])

    kf = StratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=True)
    for k, (tr_id, va_id) in enumerate(kf.split(id_mask_count['img_id'].values, id_mask_count['count'])):
        train_ids = id_mask_count['img_id'].values[tr_id]
        valid_ids = id_mask_count['img_id'].values[va_id]
        if k == fold_number: break

    #train_ids, valid_ids = train_test_split(id_mask_count['img_id'].values, random_state=1, stratify=id_mask_count['count'], test_size=0.25)
    test_ids = sub['Image_Label'].apply(lambda x: x.split('_')[0]).drop_duplicates().values

    if use_all_data: train_ids = id_mask_count['img_id'].values

    return train, sub, train_ids, valid_ids, test_ids


def draw_convex_hull(mask, mode='convex'):
    img = np.zeros(mask.shape)
    contours, hier = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        if mode == 'rect':  # simple rectangle
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), -1)
        elif mode == 'convex':  # minimum convex hull
            hull = cv2.convexHull(c)
            cv2.drawContours(img, [hull], 0, (255, 255, 255), -1)
        elif mode == 'approx':
            epsilon = 0.02 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)
            cv2.drawContours(img, [approx], 0, (255, 255, 255), -1)
        else:  # minimum area rectangle
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img, [box], 0, (255, 255, 255), -1)
    return img / 255.
    

def rle_decode(mask_rle: str = '', shape: tuple = (1400, 2100)):
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    if mask_rle is np.nan: return img.reshape(shape, order='F')
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    
    for lo, hi in zip(starts, ends): img[lo:hi] = 1
    return img.reshape(shape, order='F')

def resize_(img, dsize=(525,350)):
    img = cv2.resize(img, dsize=dsize, interpolation=cv2.INTER_LINEAR)
    return img

def resize_to_submission(img):
    img = cv2.resize(img, dsize=(525,350), interpolation=cv2.INTER_LINEAR)
    return img


def make_mask(df: pd.DataFrame, image_name: str='img.jpg', shape: tuple = (1400, 2100)):
    encoded_masks = df.loc[df['im_id'] == image_name, 'EncodedPixels']
    masks = np.zeros((shape[0], shape[1], 4), dtype=np.float32)
    for idx, label in enumerate(encoded_masks.values):
        if label is not np.nan:
            mask = rle_decode(label)
            masks[:, :, idx] = mask
    return masks

def transpose_fw(x): # tranpose forward ijk to kij
    return x.transpose(2, 0, 1).astype('float32')

def transpose_bw(x): # tranpose ijk to jki
    return x.transpose(1, 2, 0).astype('float32')

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def mask2rle(img):
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

sigmoid = lambda x: 1 / (1 + np.exp(-x))

def scale_contour(cnt, scale):
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    cnt_norm = cnt - [cx, cy]
    cnt_scaled = cnt_norm * scale
    cnt_scaled = cnt_scaled + [cx, cy]
    cnt_scaled = cnt_scaled.astype(np.int32)
    return cnt_scaled


def scale_all_contours(mask, p=0.2, scale_min=0.8, scale_max=1.2):
    _, thresh = cv2.threshold(mask, 0.5, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        prob = np.random.uniform(low=0, high=1)
        if prob > p:
            scale = np.random.choice([0.7, 0.8, 0.9, 1.1, 1.2, 1.3])
            cnt_scaled = scale_contour(contours[i], scale)
            cv2.fillPoly(mask, pts =[cnt_scaled], color=(255,255,255))
    mask[mask>=1] = 1
    return mask

def post_process(pred, threshold, min_size):
    """
    Post processing of each predicted mask, components with fewer pixels
    than `min_size` are ignored
    """
    pred = cv2.threshold(pred, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(pred.astype(np.uint8))
    pred_mask = np.zeros((350, 525), np.float32)
    #print(num_component)
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size: pred_mask[p] = 1
    return pred_mask
    

def adv_post_process(pred, thresholds, min_sizes):
    pred0 = cv2.threshold(pred, thresholds[0], 1, cv2.THRESH_BINARY)[1]
    pred1 = cv2.threshold(pred, thresholds[1], 1, cv2.THRESH_BINARY)[1]
    num_component0, component0 = cv2.connectedComponents(pred0.astype(np.uint8))
    num_component1, component1 = cv2.connectedComponents(pred1.astype(np.uint8))

    pred_mask0 = np.zeros((350, 525), np.float32)
    for c in range(1, num_component0):
        p = (component0 == c)
        if p.sum() > min_sizes[0]: pred_mask0[p] = 1

    pred_mask1 = np.zeros((350, 525), np.float32)
    for c in range(1, num_component1):
        p = (component1 == c)
        if p.sum() > min_sizes[1]: pred_mask1[p] = 1

    pred_mask = np.logical_or(pred_mask0, pred_mask1).astype(np.uint8)
    return pred_mask

def dice(img1, img2):
    img1 = np.asarray(img1).astype(np.bool)
    img2 = np.asarray(img2).astype(np.bool)
    if img1.sum() + img2.sum() == 0: return 1
    intersection = np.logical_and(img1, img2)
    return 2. * intersection.sum() / (img1.sum() + img2.sum())


class CloudDataset(Dataset):
    def __init__(self, df, datatype, img_ids, path, img_size, resize_mask=False):
        self.df = df
        self.data_folder = f"{path}/train_images" if datatype != 'test' else f"{path}/test_images"
        self.img_ids = img_ids
        self.resize_mask = resize_mask
        self.datatype = datatype
        self.img_size = img_size
        self.augment = Compose([
            Flip(p=0.5),
            ShiftScaleRotate(scale_limit=0.3, rotate_limit=180, shift_limit=0.2, p=0.1, border_mode=0),
            OneOf([GridDistortion(), OpticalDistortion(), ElasticTransform()], p=0.1),
            OneOf([CLAHE(clip_limit=2), IAASharpen(), IAAEmboss(),RandomBrightnessContrast()], p=0.1),
            Cutout(p=0.1, num_holes=2, max_h_size=50, max_w_size=50),
            RandomCrop(p=0.2, height=1000, width=1700),
            Resize(img_size[0], img_size[1]),
            ])
        self.preprocess = Compose([
            Lambda(image=smp.encoders.get_preprocessing_fn('se_resnext50_32x4d', 'imagenet')) ,
            Lambda(image=to_tensor, mask=to_tensor)])


    def __getitem__(self, idx):
        # Load images
        image_name = self.img_ids[idx]
        mask = make_mask(self.df, image_name)
        image_path = os.path.join(self.data_folder, image_name)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # TTA for test
        if self.datatype in ['valid', 'test']:
            imgs_tta = []
            TTA_list = [ Compose([HorizontalFlip(p=1), Resize(self.img_size[0], self.img_size[1])]),
                Compose([VerticalFlip(p=1), Resize(self.img_size[0], self.img_size[1])]),
                Compose([Resize(self.img_size[0], self.img_size[1])]) 
                ]

            for TTA in TTA_list:
                tta = TTA(image=img, mask=mask)
                img_aug, mask_aug = tta['image'], tta['mask']
                preprocessed = self.preprocess(image=img_aug, mask=mask_aug)
                img_aug, mask_aug =  preprocessed['image'], preprocessed['mask']
                imgs_tta.append(img_aug)

            mask = mask_aug # use the last TTA item (original, just resize) to get the resized mask
            img = imgs_tta

        if self.datatype == 'train':
            augmented = self.augment(image=img, mask=mask)
            img, mask = augmented['image'], augmented['mask']
            preprocessed = self.preprocess(image=img, mask=mask)
            img, mask =  preprocessed['image'], preprocessed['mask']


        # Scale masks (augmentation)
        if  self.resize_mask:
            for j, m in enumerate(mask):
                m_aug = scale_all_contours(m)
                mask[j] = m_aug

        # The below 5 lines are for ProbabilisticUnet only (remove when using DeepLabV3)
        # img = img.astype(float)
        # mask = mask[0] # choose 1 class: Sugar or Flower or....
        # mask = mask.astype(float)
        # img = torch.from_numpy(img).type(torch.FloatTensor)
        # mask = torch.from_numpy(mask).type(torch.FloatTensor)

        return img, mask, image_name

    def __len__(self):
        return len(self.img_ids)


def train_one_epoch(mdl, mdl_type, loader, optimizer, loss_fn, device, accumulation_steps=1):
    start = time.time()
    mdl.train()
    total_loss = 0

    for i, (images, masks, _) in enumerate(tqdm(loader)):
        #if i == 10: break

        y_preds = mdl(images.to(device))
        if mdl_type=='deeplab': y_preds = y_preds['out']
        loss = loss_fn(y_preds, masks.to(device))
        total_loss += loss.cpu().detach().numpy()
        loss /= accumulation_steps
        loss.backward()
        if (i+1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        if (i+1)%100 == 0: print('train loss after', i+1, 'batches:', total_loss/(i+1))
    
    end = time.time()
    return np.round(total_loss / len(loader), 5), timer(start, end)


def validate_one_epoch(mdl, mdl_type, loader, device, use_max=False):
    start = time.time()
    mdl.eval()
    CLOUDTYPES = ['Fish', 'Flower', 'Gravel', 'Sugar']
    THRES = np.arange(30, 100, 10)
    MINSIZE = [10000, 12500, 15000, 17500, 20000, 22500, 25000]

    cols = [cloudtype+'_'+str(thres)+'_'+str(minsize) for cloudtype in CLOUDTYPES
       for thres in THRES for minsize in MINSIZE]
    valid_loss = pd.DataFrame(columns=cols)
    valid_loss.loc[0] = 0

    for j, (images, masks, _) in enumerate(tqdm(loader)):
        #if j == 10: break
        with torch.no_grad():
            # TTA
            if mdl_type != 'deeplab':
                y_hor = mdl(images[0].to(device)).cpu().detach().numpy()
                y_ver = mdl(images[1].to(device)).cpu().detach().numpy()
                y_ori = mdl(images[2].to(device)).cpu().detach().numpy()
            else:
                y_hor = mdl(images[0].to(device))['out'].cpu().detach().numpy()
                y_ver = mdl(images[1].to(device))['out'].cpu().detach().numpy()
                y_ori = mdl(images[2].to(device))['out'].cpu().detach().numpy()
            ###
            masks = masks.cpu().detach().numpy()
            # print('y_ver[0]', y_ver[0].shape)
            # print('masks[0]', masks[0].shape)

            # Inverse TTA
            TTA = Compose([VerticalFlip(p=1)])
            y_ver = TTA(image=transpose_bw(y_ver[0]), mask=transpose_bw(masks[0]))['image']
            y_ver = transpose_fw(y_ver)

            TTA = Compose([HorizontalFlip(p=1)])
            y_hor = TTA(image=transpose_bw(y_hor[0]), mask=transpose_bw(masks[0]))['image']
            y_hor = transpose_fw(y_hor)
            
            # Average TTA results
            y_preds = [(y_ori[0] + y_hor + y_ver) / 3]
            #y_preds = [y_ori[0]]


            for img, mask in zip(y_preds, masks):
                img_sigmoid_resized = np.zeros((4, 350, 525))
                mask_resized = np.zeros((4, 350, 525))

                for j, (img_layer, mask_layer) in enumerate(zip(img, mask)):
                    img_sigmoid_resized[j] = sigmoid(resize_to_submission(img_layer))
                    mask_resized[j] = resize_to_submission(mask_layer)

                if use_max:
                    for j, img_layer in enumerate(img_sigmoid_resized):
                        for m in range(350):
                            for n in range(525):
                                if img_sigmoid_resized[j, m, n] < np.max(img_sigmoid_resized[:, m, n]):
                                    img_sigmoid_resized[j, m, n] = 0 

                for j, (img_layer, mask_layer) in enumerate(zip(img_sigmoid_resized, mask_resized)):
                    for thres in THRES:
                        for minsize in MINSIZE:
                            img_layer_postprocessed = post_process(img_layer, thres/100, minsize)
                            col = CLOUDTYPES[j]+'_'+str(thres)+'_'+str(minsize)
                            valid_loss[col] += dice(img_layer_postprocessed, mask_layer) / loader.batch_size

    valid_loss /= len(loader)
    end = time.time()
    return valid_loss,  timer(start, end)

def get_lr(optimizer):
    return [group['lr'] for group in optimizer.param_groups][0]

def make_prediction(mdl, mdl_type, loader, device, postprocess_params, use_max=False):
    start = time.time()
    mdl.eval()
    encoded_pixels = []
    all_preds = np.zeros((len(loader), 4, 350, 525), dtype=np.int8)
    all_imgs_name = []
    cnt = 0
    for images, masks, imgs_name in tqdm(loader):
        with torch.no_grad():
            # TTA
            if mdl_type != 'deeplab':
                y_hor = mdl(images[0].to(device)).cpu().detach().numpy()
                y_ver = mdl(images[1].to(device)).cpu().detach().numpy()
                y_ori = mdl(images[2].to(device)).cpu().detach().numpy()
            else:
                y_hor = mdl(images[0].to(device))['out'].cpu().detach().numpy()
                y_ver = mdl(images[1].to(device))['out'].cpu().detach().numpy()
                y_ori = mdl(images[2].to(device))['out'].cpu().detach().numpy()
            ###
            masks = masks.cpu().detach().numpy()
            
            # Inverse TTA
            TTA = Compose([VerticalFlip(p=1)])
            y_ver = TTA(image=transpose_bw(y_ver[0]), mask=transpose_bw(masks[0]))['image']
            y_ver = transpose_fw(y_ver)

            TTA = Compose([HorizontalFlip(p=1)])
            y_hor = TTA(image=transpose_bw(y_hor[0]), mask=transpose_bw(masks[0]))['image']
            y_hor = transpose_fw(y_hor)
            # Average TTA results
            y_preds = [(y_ori[0] + y_hor + y_ver) / 3]
            #y_preds = [y_ori[0]]
            
            for img, img_name in zip(y_preds, imgs_name):
                
                img_layer_sigmoid = np.zeros((4, 350, 525))
                # Get sigmoid and resize
                for j, img_layer in enumerate(img):
                    img_layer_sigmoid[j] = sigmoid(resize_to_submission(img_layer))
                all_preds[cnt] = (img_layer_sigmoid*100).astype(np.int8)
                all_imgs_name.append(img_name)
                cnt += 1

                # Each pixel only keep sigmoid probability of the max class, otherwise 0
                if use_max:
                    for j, img_layer in enumerate(img_layer_sigmoid):
                        for m in range(350):
                            for n in range(525):
                                if img_layer_sigmoid[j, m, n] < np.max(img_layer_sigmoid[:, m, n]):
                                    img_layer_sigmoid[j, m, n] = 0 
                
                # Postprocess threshold and minsize
                for j, img_layer in enumerate(img_layer_sigmoid):
                    thres, minsize = postprocess_params['thres'][j], postprocess_params['minsize'][j]
                    img_layer_postprocessed = post_process(img_layer, thres/100, minsize)
                    encoded_pixels.append(mask2rle(img_layer_postprocessed))

    end = time.time()

    return encoded_pixels, timer(start, end), all_preds, all_imgs_name