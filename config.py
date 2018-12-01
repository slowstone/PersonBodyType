import json
import os

class Config(object):
    def __init__(self,mode='train'):
        self.mode = mode
        self.param = {}
        self.param['ARCHITECTURE'] = 'res50'
        # res50 vgg16 vgg19 incepv3 xception
        self.param['INPUT_SHAPE'] = (512,256,3)
        # h w c
        self.param['MODEL_NAME'] = 'regress'
        self.param['CLASS_NUMS'] = 4
        self.param['BETA_NUMS'] = 10
        self.param['POSE_NUMS'] = 72
        self.param['MODEL_PATH'] = './logs/regress_up_20181127T2309/ep_0004.h5'
        
        if mode == 'train':
            self.train_init()
    
    def train_init(self):
        self.param['TRAIN_FROM'] = 'res4+'
        """
        res50: all,res3+,res4+,res5+,head
        vgg16:all,head
        vgg19:all,head
        other network hadn't test
        """
        """
        # GPU Titan X 16G
        # 60 for head  in res50
        # 50 for res4+
        # 40 for res3+
        """
        self.param['IM_NAME'] = 'up-3d-box'
        self.param['DATA_VERSION'] = 'up'
        """
        image_women: origin image
        split: split by mask_rcnn
        test_im: for test
        up-3d: 3d parts of up dataset
        up-3d-box: cut up-3d by box
        up-3d-mask: change up-3d(RGB) to mask (01 matrix)(one channel copy to three channel)
        up-3d-mask-box: cut up-3d mask by box (01 matrix)(one channel copy to three channel)
        """
        """
        v1: bmi im_women
        v2: shape im_women
        v3: shape im_women(Manual screening)
        v4: shape split
        v5: bmi split
        up: SMPL up-3d using for im_name with up-3d*
        """
        self.param['BATCH_SIZE'] = 50
        """
        # adam, momentum or nesterov
        """
        self.param['OPT_STRING'] = "adam"

        self.param['LEARNING_RATE'] = 0.001
        #using in lr decay
        self.param['LR_DECAY'] = 0.95
        #using in momentum and nesterov
        self.param['MOMENTUM'] = 0.9
        #using in adam
        self.param['ADAM_BETA_1'] = 0.9
        self.param['ADAM_BETA_2'] = 0.99

        self.param['L2_SCALE'] = 0.01

        #base hyper-parameter
        self.param['MEAN_PIXEL'] = [93.2,104.6,116.6]

        
        self.param['TRAIN_STEPS'] = 500
        self.param['VALIDATION_STEPS'] = 50
        self.param['EPOCHS'] = 20
        
    def show_config(self):
        print("\n============== Param =============")
        print("===========>config mode",self.mode)
        for key in self.param.keys():
            print(key,":",self.param[key])
        print("==================================\n")
    
    def set_config(self,file_path):
        print("\n======set config from file=======")
        print(file_path)
        f = open(file_path,'r')
        json_infos = json.load(f)
        for key in self.param:
            self.param[key] = json_infos[key]
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