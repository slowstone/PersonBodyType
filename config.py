import json
import os

class Config(object):
    def __init__(self):
        self.param = {}
        self.param['ARCHITECTURE'] = 'res50'
        """
        # adam, momentum or nesterov
        """
        self.param['OPT_STRING'] = "adam"

        self.param['LEARNING_RATE'] = 0.01
        #using in lr decay
        self.param['LR_DECAY'] = 0.95
        #using in momentum and nesterov
        self.param['MOMENTUM'] = 0.9
        #using in adam
        self.param['ADAM_BETA_1'] = 0.9
        self.param['ADAM_BETA_2'] = 0.99

        self.param['L2_SCALE'] = 0.1

        #base hyper-parameter
        self.param['INPUT_SHAPE'] = (512,256,3)
        self.param['MEAN_PIXEL'] = [93.2,104.6,116.6]

        """
        # GPU Titan X 16G
        # 60 for head
        # 50 for 4+
        # 40 for 3+
        """
        self.param['TRAIN_FROM'] = '4+'
        self.param['BATCH_SIZE'] = 50
        self.param['TRAIN_STEPS'] = 500
        self.param['VALIDATION_STEPS'] = 50
        self.param['EPOCHS'] = 200
        self.param['CLASS_NUMS'] = 4
        self.param['BETA_NUMS'] = 10
        self.param['POSE_NUMS'] = 72
        
        self.param['MODEL_PATH'] = None
#         self.param['MODEL_PATH'] = './logs/regress_up_20181125T1108/ep_0004.h5'

        self.param['MODEL_NAME'] = 'regress'
        self.param['IM_NAME'] = 'up-3d'
        self.param['DATA_VERSION'] = 'up'
        # model_name = 'regress'
        # im_name = 'up-3d'
        # data_version = 'up'
        """
        image_women: origin image
        split: split by mask_rcnn
        up-3d: 3d parts of up dataset
        test_im: for test
        """
        """
        v1: bmi im_women
        v2: shape im_women
        v3: shape im_women(Manual screening)
        v4: shape split
        v5: bmi split
        up: SMPL up-3d
        """
    
    def show_config(self):
        print("\n============== Param =============")
        for key in self.param.keys():
            print(key,":",self.param[key])
        print("==================================\n")
    
    def save_config(self,out_dir = './',out_name = 'config.json'):
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        out_path = os.path.join(out_dir,out_name)
        f = open(out_path,'w')
        f.write(json.dumps(self.param,indent=2))
    
    def set_param(self,keys,datas):
        for i,key in enumerate(keys):
            self.param[key] = datas[i]
            print("set",key,"to",datas[i])