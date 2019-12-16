import efficientnet.keras as efn
from keras.layers import Dense
from keras.models import Model
import keras.backend as K

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