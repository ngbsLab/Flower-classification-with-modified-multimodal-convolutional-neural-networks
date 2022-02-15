import os
import shutil
from abc import ABC, abstractmethod
import random
from dataset.oxfordflower.data_utils import *
from dataset.oxfordflower.tf_utils import *
import tensorflow as tf
from collections import defaultdict
from glob import glob
from PIL import Image
from dataset.data_generator import Database

class OxfordFlower(Database):
    """
    Database for OxfordFlower classification.
    Preprocess raw dataset into tfrecords, in which (image,text) pairs are saved as bytes string.
    If tfrecords already exist, initializing(getting an instace) does nothing.
    
    NOTE) Utility functions for processing text,image are in 'dataset.oxfordflower.data_utils' and 'dataset.oxfordflower.tf_utils'
          
    """   
    def __init__(self,config_path=None,seed=1234,ratio=(0.8,0.1,0.1)):

        self.config = parse_config(config_path)
        self.config["seed"]= seed
        
        self.raw_database_address = self.config["base_path"]
        self.database_address = self.config["data_path"]
        
        self.train_dict, self.eval_dict, self.test_dict = self.get_train_val_test_folders(seed,ratio)
        
        self.prepare_database()
        
        self.input_shape = self.get_input_shape() 
                
    def prepare_database(self):
        """
        All results are saved in a directory,
        """
        config = self.config
        
        # argparser로 is_word2vec, additional_corpus할지 결정
        if not os.path.exists(config["data_path"]):
            os.makedirs(config["data_path"],exist_ok=True)
        if not os.path.exists(os.path.join(config["data_path"],config["tfrecord_path"])):
            os.makedirs(os.path.join(config["data_path"],config["tfrecord_path"]),exist_ok=True)
        if not os.path.exists(os.path.join(config["data_path"],config["pretrain_path"])):
            os.makedirs(os.path.join(config["data_path"],config["pretrain_path"]))
        
        # Load train,eval,test dictionary
        train_dict = load_json(os.path.join(config["base_path"],config["train_json"]))
        eval_dict = load_json(os.path.join(config["base_path"],config["eval_json"]))
        test_dict = load_json(os.path.join(config["base_path"],config["test_json"]))
        _tokenizer = tokenizer("Okt") # argparser로 줄것 
        
        # build and save vocab
        vocab=build_vocab(config,_tokenizer)
        with open(join(config["data_path"],config["vocab"],"wb") as f:
            pickle.dump(vocab,f)
        
        # Tokenize according to a vocab
        name2token = txt2Token(config,_tokenizer,vocab)
        
        # Encode images to raw byte string
        name2img = img2Raw(config)
        
        # Creator TFRecordWriter
        train_tfwriter=tf.io.TFRecordWriter(
            join(config["data_path"],config["tfrecord_path"],"train.record"))
        eval_tfwriter=tf.io.TFRecordWriter(
            join(config["data_path"],config["tfrecord_path"],"eval.record"))
        test_tfwriter=tf.io.TFRecordWriter(
            join(config["data_path"],config["tfrecord_path"],"test.record"))    
        
        # Write according to train,eval,test dictionary
        for _id, example in train_dict.items():
            match_and_write(_id,example,name2img,name2token,train_tfwriter)
        train_tfwriter.close()
            
        for _id, example in eval_dict.items():
            match_and_write(_id,example,name2img,name2token,eval_tfwriter)
        eval_tfwriter.close()
            
        for _id, example in test_dict.items():
            match_and_write(_id,example,name2img,name2token,test_tfwriter)
        test_tfwriter.close()
        
    def get_train_val_test_folders(self,seed=1234,ratio=(0.8,0.1,0.1)):
        print("For multi-view training, Split the whole dataset into train,evaluation and testset, and return path of training,evaluation,test json file")
        
        img_path = os.path.join(self.raw_database_address,"images")
        txt_path = os.path.join(self.raw_database_address,"texts")
        
        # Check if data valid
        assert len(os.listdir(img_path)) == len(os.listdir(txt_path)),"Num classes differs"
        
        total_num = 0
        for fdir in os.listdir(img_path):
            total_num+=len(os.listdir(os.path.join(img_path,fdir)))

        print("Total Num :",total_num)
        
        whole_imgs = []
        for fdir in os.listdir(img_path):
            imgs = os.listdir(os.path.join(img_path,fdir))
            for img in imgs:
                whole_imgs.append((int(fdir),img))    
        whole_imgs = sorted(whole_imgs,key=lambda x:x[-1])

        whole_txts = []
        for fdir in os.listdir(txt_path):
            txts = os.listdir(os.path.join(txt_path,fdir))
            for txt in txts:
                whole_txts.append((int(fdir),txt))    
        whole_txts = sorted(whole_txts,key=lambda x:x[-1])

        whole_lst = []
        for (cls,img),(cls2,txt)in zip(whole_imgs,whole_txts):
            assert cls == cls2
        whole_lst.append((cls,img,txt))
        
        random.shuffle(whole_lst)
        
        trn_num  = int(total_num*ratio[0])
        eval_num = int(total_num*ratio[1])
        test_num = total_num - trn_num - eval_num
        assert trn_num+eval_num+test_num == total_num

        trn_dict = {}
        eval_dict = {}
        test_dict = {}

        for i,(cls,img,txt) in enumerate(whole_lst):
    
            if i < trn_num:
                trn_dict[str(i)] = {"class":cls,"img_file":img,"txt_file":txt}
            elif i < trn_num+eval_num:
                eval_dict[str(i)] = {"class":cls,"img_file":img,"txt_file":txt}
            else:
                test_dict[str(i)] = {"class":cls,"img_file":img,"txt_file":txt}

        save_json(trn_dict,os.path.join(self.config["base_path"],self.config["train_json"]))
        save_json(eval_dict,os.path.join(self.config["base_path"],self.config["eval_json"]))
        save_json(test_dict,os.path.join(self.config["base_path"],self.config["test_json"]))

        return trn_dict,eval_dict,test_dict
    
    def get_input_shape(self):
        return self.config["img_size"]
    
    def preview_image(self,img_dir):
        return Image.open(img_dir)
    
    def get_class(self):
        return self.train_dict, self.eval_dict, self.test_dict
    
    def data_loader(self,usage,mode,shuffle,batch_size):
        config = self.config
        
        _parse_img_example,_parse_txt_example,_parse_single_example = config_wrapper_parse_funcs(config)
        assert usage in ["train","eval","test"], "mode should be one of 'train','eval','test'"
        
        if mode=="text" or mode=="txt":
            parse_func = _parse_txt_example
        elif mode=="image" or mode=="img":
            parse_func = _parse_img_example
        elif mode=="both" or mode==None:
            parse_func = _parse_single_example
        else:
            parse_func = _parse_single_example
            
        path = join(config["data_path"],config["tfrecord_path"],"train.record")
        dataset = tf.data.TFRecordDataset(path).map(parse_func)
        dataset = dataset.shuffle(batch_size).batch(batch_size)
        return dataset
    
    def directory_info(self):
        """
        Print directory information.
        """
        pass
