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

        self.param['L2_SCALE'] = 0.001

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
        self.param['EPOCHS'] = 20
        self.param['CLASS_NUMS'] = 4
    
    def show_config(self):
        print("============== Param =============")
        for key in self.param.keys():
            print(key,":",self.param[key])
        print("==================================")
    
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