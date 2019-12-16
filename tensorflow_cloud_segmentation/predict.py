from sklearn.model_selection import StratifiedKFold
import numpy as np, pandas as pd, cv2, argparse, sys, pickle, os, ast
from tta_wrapper import tta_segmentation
from tqdm import tqdm
import segmentation_models as sm
from keras.optimizers import Adam
import keras.backend as K

def get_test_data(input_path):
    sub_df = pd.read_csv(input_path + 'sample_submission.csv')
    sub_df['ImageId'] = sub_df['Image_Label'].apply(lambda x: x.split('_')[0])
    test_imgs = pd.DataFrame(sub_df['ImageId'].unique(), columns=['ImageId'])
    return sub_df, test_imgs

def get_model(model,backbone,opt,loss,metric,shape):
    h,w = shape
    wei, act = 'imagenet', 'sigmoid'
    if model == 'pspnet':model = sm.PSPNet(backbone, classes=4, input_shape=(h, w, 3), activation=act,encoder_weights=wei) 
    elif model == 'fpn': model = sm.FPN(backbone,classes=4,input_shape=(h, w, 3),activation=act,encoder_weights=wei)
    elif model == 'linknet': model = sm.Linknet(backbone, classes=4,  input_shape=(h, w, 3), activation=act, encoder_weights=wei)
    elif model == 'unet': model = sm.Unet(backbone, classes=4, input_shape=(h, w, 3), activation=act,  encoder_weights=wei)
    else: raise ValueError('Unknown model specification ' + model)
    model.compile(optimizer=opt, loss=loss, metrics=[metric])
    return model

def dummy_loss(y_true, y_pred):
    return 1

def load_img(img_name,  backbone='efficientnetb3', input_path=''):
    input_path = input_path + 'resized'
    _preprocess = sm.get_preprocessing(backbone)
    img_path = f"{input_path}/{img_name}"
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)
    img = _preprocess(img)
    return img

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred = K.cast(y_pred, 'float32')
    y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')
    intersection = y_true_f * y_pred_f
    score = 2. * K.sum(intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))
    return score

def reshape_to_submission_single(pred):
    output = np.zeros((350, 525, 4), dtype=np.int16)
    for t in range(4):
        pred_layer = pred[:,:,t]
        pred_layer = cv2.resize(pred_layer, dsize=(525, 350), interpolation=cv2.INTER_LINEAR) 
        pred_layer = (pred_layer*100).astype(np.int16)
        output[:,:,t] = pred_layer
    return output


def make_prediction(model_type, backbone,  seed, n_splits, model_path, tta=True, shape=(384,576), input_path='', make_valid=True, make_test=True):
    _,test_imgs = get_test_data(input_path)
   
    train_df = pd.read_csv(input_path+'train_384x576.csv')
    train_df['rles'] = train_df['rles'].apply(ast.literal_eval)

    skf = StratifiedKFold(n_splits=n_splits,  random_state=seed, shuffle=True)

    if make_valid:
        all_preds = np.zeros((len(train_df), 350, 525, 4), dtype=np.int16)
        cnt, val_names = 0, []
        for n_fold, (_, val_indices) in enumerate(skf.split(train_df, train_df['hasMask_eachtype'])):
            print('Predicting oof for fold', n_fold, '...')
            model = get_model(model_type, backbone, Adam(), dice_coef, dice_coef, shape)
            model_fold_filepath = model_path +  str(n_fold) + '.h5'
            model.load_weights(model_fold_filepath)

            if tta: model = tta_segmentation(model, h_flip=True,h_shift=(-15, 15), merge='mean')

            for i, img_name in tqdm(enumerate(train_df.iloc[val_indices]['im_id'])):
                single_img = load_img(img_name, backbone=backbone, input_path=input_path)
                Y = model.predict(single_img)
                Y = reshape_to_submission_single(Y[0])
                all_preds[cnt] = Y
                val_names.append(img_name)
                cnt += 1

        all_preds = all_preds.astype(np.int8)
        filesave = model_path + '_oof.npy'
        np.save(model_path + '_names.npy', val_names)
        np.save(filesave, all_preds)
        print('Oof prediction file saved to', filesave)


    if make_test:
        all_preds = np.zeros((len(test_imgs), 350, 525, 4), dtype=np.int16)
        cnt, val_names = 0, []
        for n_fold, (_, val_indices) in enumerate(skf.split(train_df, train_df.hasMask)):
            print('Predicting test for fold', n_fold, '...')
            model = get_model(model_type, backbone, Adam(), dice_coef, dice_coef, shape)
            model_fold_filepath = model_path +  str(n_fold) + '.h5'
            model.load_weights(model_fold_filepath)

            if tta: model = tta_segmentation(model, h_flip=True,h_shift=(-15, 15),  merge='mean')

            for i, img_name in tqdm(enumerate(test_imgs.ImageId)):
                single_img = load_img(img_name, backbone=backbone, input_path=input_path)
                Y = model.predict(single_img)
                Y = reshape_to_submission_single(Y[0])
                all_preds[i] += Y

        all_preds = (all_preds // n_splits).astype(np.int8)
        filesave = model_path + '_test.npy'
        np.save(filesave, all_preds)
        print('Test prediction file saved to', filesave)
        

def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args(args):
    parser = argparse.ArgumentParser(description='Predict script')
    parser.add_argument('--model', help='Segmentation model', default='unet')
    parser.add_argument('--backbone', help='Model backbone', default='efficientnetb5', type=str)
    parser.add_argument('--splits', default=6, type=int)
    parser.add_argument('--seed', help='kfold seed', default=1, type=int)
    parser.add_argument('--machine', help='cibci or home', default='home', type=str)
    parser.add_argument('--modelfilename', help='filename', default='fpn_efficientnetb3_bs3_shape384x576_usecutmix1_seed1_fold', type=str)
    parser.add_argument('--make_valid',  default='True', type=str)
    parser.add_argument('--make_test',  default='True', type=str)
    return parser.parse_args(args)

if __name__ == '__main__':
    args = sys.argv[1:]
    args = parse_args(args)

    if args.machine == 'home':
        input_data_path = 'D:/cloud_kaggle_data/input/'
        model_path = 'D:/cloud_kaggle_data/trained_models/' + args.modelfilename
    elif args.machine == 'cibci':
        input_data_path = '/data/khavo/cloud_kaggle/input/'
        model_path = '/data/khavo/cloud_kaggle/trained_models/' + args.modelfilename
    else: raise ValueError('Machine specification error')


    make_prediction(model_type=args.model, backbone=args.backbone, seed=args.seed, n_splits=args.splits,
     model_path=model_path, input_path=input_data_path, make_valid=str2bool(args.make_valid), make_test=str2bool(args.make_test))

    # python predict.py --model unet --backbone efficientnetb3 --seed 2 --splits 6 
    # --modelfilename unet_efficientnetb3_bs3_shape384x576_usecutmix1_seed2_fold --machine cibci

