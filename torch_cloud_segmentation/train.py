from cloud_kaggle_lib import *

BASEPATH = '/home/khavo/Data/cloud_kaggle/'

DATAPATH = BASEPATH + 'input/' #'input_tiny/'
BATCH_SIZE = 2
ACCU_STEPS = 2
NUM_EPOCHS = 40
START_LR = 0.001
PATIENCE = 3
IMG_SIZE = [448, 672]
MODEL_TYPE = 'u-res'
LOG_FILE = BASEPATH + 'logs/tta_Uresnext50_448x672_seed2'
MODEL_FILE = BASEPATH +'models/tta_Uresnext50_448x672_seed2'
VALID_FILE = BASEPATH +'valid_preds/tta_Uresnext50_448x672_seed2'
TEST_FILE = BASEPATH +'test_preds/tta_Uresnext50_448x672_seed2'

VALIDATION_MODE = True
LOG_FREQUENCY = 1
RESIZE_MASK = False

USE_ALL_DATA = False
N_SPLITS = 6
FOLD_NUMBER = [3,4,5]
SEED = 2

# -------------------

for fold_number in FOLD_NUMBER:

  model_file = MODEL_FILE + '_fold' + str(fold_number) + '.ckpt'
  log_file = LOG_FILE + '_fold' + str(fold_number) + '.csv'
  valid_file = VALID_FILE + '_fold' + str(fold_number)
  test_file = TEST_FILE + '_fold' + str(fold_number)

  if MODEL_TYPE == 'deeplab':
  ### DeeplabV3
    ENCODER = 'resnet50'
    ENCODER_WEIGHTS = 'imagenet'
    model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=4)


  if MODEL_TYPE == 'u-res':
  ### Unet-Resnet
    ENCODER = 'se_resnext50_32x4d'#'se_resnext50_32x4d'
    ENCODER_WEIGHTS = 'imagenet'
    model = smp.Unet(
        encoder_name=ENCODER, 
        encoder_weights=ENCODER_WEIGHTS, 
        classes=4, 
        activation=None,
    )

  if MODEL_TYPE == 'u-eff':
  ## Unet-EfficientNet
    ENCODER = 'resnet50'
    ENCODER_WEIGHTS = 'imagenet'
    model = get_efficientunet_b3(out_channels=4, concat_input=True, pretrained=True)


  # -----------------------------------------------
  print('Training fold', fold_number, 'seed', SEED, '...')
  CLOUDTYPES = ['Fish', 'Flower', 'Gravel', 'Sugar']

  #from probabilistic_unet import ProbabilisticUnet
  #from utils import l2_regularisation

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  train, sub, train_ids, valid_ids, test_ids = load_csv(path=DATAPATH, n_splits=N_SPLITS, seed=SEED, fold_number=fold_number, use_all_data=USE_ALL_DATA)
  model.to(device)

  params = [p for p in model.parameters() if p.requires_grad]
#  optimizer = torch.optim.Adam(params, lr=START_LR, momentum=0.9, weight_decay=0.0005)

  optimizer = torch.optim.Adam([
      {'params': model.encoder.parameters(), 'lr': START_LR},
      {'params': model.decoder.parameters(), 'lr': 0.01},
  ])

  lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=2)
  #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.15)
  # Remember to change lr_scheduler.step(val_loss) or .step() depending on lr_scheduler type


  early_stopping = EarlyStopping(patience=PATIENCE)

  
  train_dataset = CloudDataset(df=train, datatype='train', img_ids=train_ids, img_size=IMG_SIZE, path=DATAPATH, resize_mask=RESIZE_MASK)
  valid_dataset = CloudDataset(df=train, datatype='valid', img_ids=valid_ids, img_size=IMG_SIZE,  path=DATAPATH)
  test_dataset = CloudDataset(df=sub, datatype='test', img_ids=test_ids, img_size=IMG_SIZE, path=DATAPATH)

  print('Training data:', len(train_ids))
  print('Validation data:', len(valid_ids))

  train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True) #num_workers=num_workers)
  valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
  test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

  loss_func = smp.utils.losses.BCEDiceLoss(eps=1.)

  selected_fields = [cloudtype+'_'+thr+'_15000' for cloudtype in CLOUDTYPES for thr in ['30','40','50','60','70','80','90'] ]
  start_time = time.time()


  for epoch in range(NUM_EPOCHS):
      print('Training epoch', epoch, 'learning rate:', get_lr(optimizer), '.......')
      log_df = pd.DataFrame(columns=['lr','train_loss', 'val_loss', 'train_elapsed', 'val_elapsed', 'total_elapsed'],
                      data=[[0,0,0,0,0,0]])
      
      # Train
      loss_tr, elapsed_tr = train_one_epoch(model, MODEL_TYPE, train_loader, optimizer,
                                  loss_func, device, accumulation_steps=ACCU_STEPS)

      print('Train loss epoch', epoch, loss_tr, 'train time', elapsed_tr, 'lr', get_lr(optimizer))

      # Save model
      torch.save(model.state_dict(), model_file)

      # Validation and Logging 
      if VALIDATION_MODE:
        if (epoch+1)%LOG_FREQUENCY==0:
          loss_va, elapsed_va = validate_one_epoch(model, MODEL_TYPE, valid_loader, device)
          val_loss = loss_va[selected_fields].values.mean()
          end_time = time.time()
          total_elapsed = timer(start_time, end_time)
          log_df.iloc[0] = [get_lr(optimizer), loss_tr, val_loss, elapsed_tr, elapsed_va, total_elapsed]
          new_log_row = pd.concat([log_df, loss_va], axis=1)
          all_log_df = new_log_row if epoch==0 else pd.concat([all_log_df, new_log_row]).reset_index(drop=True)
          all_log_df.to_csv(log_file, index=None)
          print('val_loss:', val_loss, ', total_elapsed', total_elapsed)
          #for f in selected_fields: print(f+ ': ' + str(np.round(loss_va[f].values[0], 5)))
          print('\n')

        should_stop = early_stopping.step(1-val_loss) # because val_loss = dice is increasing
        if should_stop:
            print('Early stopping at {}'.format(epoch))
            break
      
      # Scheduler step and Early Stopping Check
      lr_scheduler.step(val_loss)
      #lr_scheduler.step()


  # Make prediction for validation
  print('Making validation predictions.....')
  log = pd.read_csv(log_file) 
  postprocess_params = get_best_stats(log)
  _, pred_time, preds, img_names = make_prediction(model, MODEL_TYPE, valid_loader, device, postprocess_params, use_max=False)    
  np.save(valid_file + '.npy', preds)
  print('Validation prediction saved to', valid_file + '.npy')
  np.save(valid_file + '_ImgsName.npy', img_names)
  print('Validation images names saved to', valid_file + '_ImgsName.npy')
  print('Making validation prediction takes', pred_time)
  

  # Make prediction for test
  print('Making test predictions.....')
  encoded_pixels, pred_time, preds, _ = make_prediction(model, MODEL_TYPE, test_loader, device, postprocess_params, use_max=False)    
  np.save(test_file + '.npy', preds)
  print('Test prediction saved to', test_file + '.npy')

  sub['EncodedPixels'] = encoded_pixels
  sub[['Image_Label', 'EncodedPixels']].to_csv( test_file + '_sub.csv', index=None)
  print('Test submission saved to', 'sub_' + test_file + '_sub.csv')
  print('Making test prediction takes', pred_time)
  print('Best stats:', postprocess_params)
  print('\n\n')
