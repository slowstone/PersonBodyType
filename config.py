import json
import os

class Config(object):
    def __init__(self,mode='train'):
        self.mode = mode
        self.param = {}
        self.param['ARCHITECTURE'] = 'res50'
        # res50 vgg16 vgg19 incepv3 xception
        self.param['INPUT_SHAPE'] = (240,240,3)
        # h w c
        self.param['MODEL_NAME'] = 'regress'
        self.param['CLASS_NUMS'] = 4
        self.param['BETA_NUMS'] = 1
        self.param['POSE_NUMS'] = 72
#         self.param['MODEL_PATH'] = './logs/regress_up_20181205T0934/ep_0100.h5'
#         self.param['MODEL_PATH'] = './logs/regress_surreal-human_20190104T1915/ep_0007.h5'
        self.param['MODEL_PATH'] = None

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
        self.param['BATCH_SIZE'] = 50
        """
        # GPU Titan X 16G
        # 60 for head  in res50
        # 50 for res4+
        # 40 for res3+
        """
        self.param['IM_NAME'] = 'image_women_human'
        self.param['DATA_VERSION'] = 'v6'
        """
        image_women: origin image
        split: split by mask_rcnn
        test_im: for test
        image_women_human: cut image by mask
        image_women_mask: mask for image_women
        up-3d: 3d parts of up dataset
        up-3d-box: cut up-3d by box
        up-3d-mask: change up-3d(RGB) to mask (01 matrix)(one channel copy to three channel)
        up-3d-mask-box: cut up-3d mask by box (01 matrix)(one channel copy to three channel)
        surreal: surreal dataset
        surreal-human: cut up-3d mask by box (01 matrix)(one channel copy to three channel)
        """
        """
        v1: bmi im_women                      use IM_NAME = image_women
        v2: shape im_women                     use IM_NAME = image_women
        v3: shape im_women(Manual screening)         use IM_NAME = image_women
        v4: shape split                       use IM_NAME = split
        v5: bmi split                        use IM_NAME = split
        v6: bmi human                        use IM_NAME = image_women_human
        v7: shape human                       use IM_NAME = image_women_human
        up: SMPL up-3d using for im_name with up-3d*    use IM_NAME = up-3d*
        surreal: just for record                 use train_surreal.py
        surreal-human: just for record             use train_surreal.py
        """

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

        self.param['L2_SCALE'] = 0.01

        #base hyper-parameter
        self.param['MEAN_PIXEL'] = [93.2,104.6,116.6]


        self.param['TRAIN_STEPS'] = 10000
        self.param['VALIDATION_STEPS'] = 1000
        self.param['EPOCHS'] = 100

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
