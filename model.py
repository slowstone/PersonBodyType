import re
import multiprocessing
import tensorflow as tf
import numpy as np
from config import Config
from dataset import imread

L2_SCALE = 0.01

def l2_reg_cate_loss(y_true, y_pred):
    regularizer = tf.contrib.layers.l2_regularizer(scale=L2_SCALE)
    reg_loss = tf.contrib.layers.apply_regularization(regularizer)
    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    all_loss = loss + reg_loss
    return all_loss

def l2_reg_mean_squ_loss(y_true, y_pred):
    regularizer = tf.contrib.layers.l2_regularizer(scale=L2_SCALE)
    reg_loss = tf.contrib.layers.apply_regularization(regularizer)
    loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
    all_loss = loss + reg_loss
    return all_loss

def myloss(y_true,y_pred):
    return tf.keras.losses.mean_squared_error(y_true,y_pred)

class Model(object):
    def __init__(self,config=Config()):
        self.config = config
        if config.mode == 'train':
            L2_SCALE = self.config.param['L2_SCALE']
            if self.config.param['MODEL_NAME'] == 'classify':
                self.model = self.build_classify_model()
            if self.config.param['MODEL_NAME'] == 'regress':
                self.model = self.build_regress_model()
            for w in self.model.trainable_weights:
                if 'kernel' in w.name:
                    tf.add_to_collection(tf.GraphKeys.WEIGHTS,w)
            if self.config.param['MODEL_PATH'] is not None:
                self.model.load_weights(self.config.param['MODEL_PATH'],by_name=True)
            self.set_trainable()
            if self.config.param['MODEL_NAME'] == 'classify':
                loss = l2_reg_cate_loss
                metrics = ['categorical_accuracy','categorical_crossentropy']
            if self.config.param['MODEL_NAME'] == 'regress':
                metrics = ['mse','mae','mape']
            self.compile_fuc(loss=myloss,metrics=metrics)
        if config.mode == 'eval':
            if self.config.param['MODEL_NAME'] == 'classify':
                self.model = self.build_classify_model()
            if self.config.param['MODEL_NAME'] == 'regress':
                self.model = self.build_regress_model()
            if self.config.param['MODEL_PATH'] is not None:
                self.model.load_weights(self.config.param['MODEL_PATH'],by_name=True)
        
    def build_classify_model(self):
        architecture = self.config.param['ARCHITECTURE']
        if architecture == 'res50':
            base_model = tf.keras.applications.resnet50.ResNet50(include_top=False,
                            weights='imagenet',input_shape=self.config.param['INPUT_SHAPE'],pooling='avg')
        if architecture == 'xception':
            base_model = tf.keras.applications.xception.Xception(include_top=False,
                            weights='imagenet',input_shape=self.config.param['INPUT_SHAPE'],pooling='avg')
        if architecture == 'vgg16':
            base_model = tf.keras.applications.vgg16.VGG16(include_top=False,
                            weights='imagenet',input_shape=self.config.param['INPUT_SHAPE'],pooling='avg')
        if architecture == 'vgg19':
            base_model = tf.keras.applications.vgg19.VGG19(include_top=False,
                            weights='imagenet',input_shape=self.config.param['INPUT_SHAPE'],pooling='avg')
        if architecture == 'incepv3':
            base_model = tf.keras.applications.inception_v3.InceptionV3(include_top=False,
                            weights='imagenet',input_shape=self.config.param['INPUT_SHAPE'],pooling='avg')
        base_model_output = base_model.output

        base_model_input = base_model.input

        output = tf.keras.layers.Dense(self.config.param['CLASS_NUMS'],activation='softmax',name='fc_softmax')(base_model_output)

        sm_model = tf.keras.models.Model(inputs = base_model_input,outputs = output, name = architecture+'_softmax')

        return sm_model

    def build_regress_model(self):
        architecture = self.config.param['ARCHITECTURE']
        if architecture == 'res50':
            base_model = tf.keras.applications.resnet50.ResNet50(include_top=False,
                            weights='imagenet',input_shape=self.config.param['INPUT_SHAPE'],pooling='avg')
        if architecture == 'xception':
            base_model = tf.keras.applications.xception.Xception(include_top=False,
                            weights='imagenet',input_shape=self.config.param['INPUT_SHAPE'],pooling='avg')
        if architecture == 'vgg16':
            base_model = tf.keras.applications.vgg16.VGG16(include_top=False,
                            weights='imagenet',input_shape=self.config.param['INPUT_SHAPE'],pooling='avg')
        if architecture == 'vgg19':
            base_model = tf.keras.applications.vgg19.VGG19(include_top=False,
                            weights='imagenet',input_shape=self.config.param['INPUT_SHAPE'],pooling='avg')
        if architecture == 'incepv3':
            base_model = tf.keras.applications.inception_v3.InceptionV3(include_top=False,
                            weights='imagenet',input_shape=self.config.param['INPUT_SHAPE'],pooling='avg')
        base_model_output = base_model.output

        base_model_input = base_model.input

        output = tf.keras.layers.Dense(self.config.param['BETA_NUMS'],activation='linear',name='fc_linear')(base_model_output)

        sm_model = tf.keras.models.Model(inputs = base_model_input,outputs = output, name = architecture+'_linear')

        return sm_model
    
    def set_trainable(self,verbose=0):
        layer_dict = {
            # all layers but the backbone
            "heads": r"(fc.*)",
            # From a specific Resnet stage and up
            "res3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(fc.*)",
            "res4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(fc.*)",
            "res5+": r"(res5.*)|(bn5.*)|(fc.*)",
            # All layers
            "all": ".*",
            }
        print("============> train from",self.config.param['TRAIN_FROM'])
        pattern = layer_dict[self.config.param['TRAIN_FROM']]
        self.resnet_set_trainable(pattern,self.model,verbose)

    def resnet_set_trainable(self,pattern,model=None,verbose=1):
        """Sets model layers as trainable if their names match
        the given regular expression.
        """
        if model is None:
            model = self.model
        if verbose > 0:
            print("\n==========>In:",model.name)
            print("===========>The trainable layers:")
        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        layers = model.layers
        for layer in layers:
            # Is the layer a model?
            if layer.__class__.__name__ == 'Model':
                resnet_set_trainable(
                    pattern, model=layer)
                continue

            if not layer.weights:
                continue
            # Is it trainable?
            trainable = bool(re.fullmatch(pattern, layer.name))
            # Update layer. If layer is a container, update inner layer.
            layer.trainable = trainable
            # Print trainble layer names
            if trainable and verbose > 0:
                print(" ",layer.name)

    def compile_fuc(self,loss,metrics=None):
        if self.config.param['OPT_STRING'] == "momentum":
            opt = tf.keras.optimizers.SGD(lr=self.config.param['LEARNING_RATE'],
                                momentum=self.config.param['MOMENTUM'])
        if self.config.param['OPT_STRING'] == "nesterov":
            opt = tf.keras.optimizers.SGD(lr=self.config.param['LEARNING_RATE'],
                                momentum=self.config.param['MOMENTUM'],
                                nesterov=True)
        if self.config.param['OPT_STRING'] == "adam":
            opt = tf.keras.optimizers.Adam(lr=self.config.param['LEARNING_RATE'],
                                beta_1=self.config.param['ADAM_BETA_1'],
                                beta_2=self.config.param['ADAM_BETA_2'])
        self.model.compile(optimizer=opt,loss=loss,metrics=metrics)

    def train(self,train_generator,val_generator,callbacks):
        self.model.fit_generator(train_generator,
                   steps_per_epoch = self.config.param['TRAIN_STEPS'],
                   epochs = self.config.param['EPOCHS'],
                   validation_data=val_generator,
                   validation_steps=self.config.param['VALIDATION_STEPS'],
                   callbacks=callbacks,
                   workers = multiprocessing.cpu_count(),
                   max_queue_size = 10,
                   shuffle = True,
                   use_multiprocessing = True)

    def eval_one_image(self,path):
        im = imread(path,self.config.param['INPUT_SHAPE'])
        if im is None:
            return im,None
        im = np.array([im])
        res = self.model.predict(im)
        return im[0],res[0]

    def eval_dir(self,im_dir):
        import os
        im_names = os.listdir(im_dir)
        outputs = {}
        for i,name in enumerate(im_names):
            im_path = os.path.join(im_dir,name)
            im,res = self.eval_one_image(im_path)
            outputs[name] = {}
            outputs[name]['res'] = res
            outputs[name]['im'] = im
            print(i+1,'/',len(im_names),end='\r')
        return outputs
