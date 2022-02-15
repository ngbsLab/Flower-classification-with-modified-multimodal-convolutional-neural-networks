from PIL import Image
import os,pickle,re,configparser
from os.path import join,exists
from collections import defaultdict,Counter
import tensorflow as tf
import numpy as np
from konlpy.tag import Okt
from tqdm import tqdm
import json
from gensim.models import Word2Vec
from dataset.oxfordflower.tf_utils import _bytes_feature,_float_feature,_int64_feature, config_wrapper_parse_funcs

args_path = "dataset/oxfordfloewr/args.ini"

def parse_config(fname=None,section="data"):
    if fname is None:
        fname = "args.ini"
    
    int_vals = ["num_class","max_len","min_freq","embed_dim","num_filters","vocab_size","seed"]
    str_list_vals = ["stop_pos","stop_words"]
    int_list_vals = ["img_size","filter_sizes"]
    float_vals = ["lambda","drop_out","drop_out2"]
    config=configparser.ConfigParser()
    config.read(fname)
    
    def as_list(inp,dtype=str,sep=","):
        _lst = inp.strip().split(sep)
        return list(map(dtype,_lst))
    
    return_dict = {}
    
    for sect in config.sections():
        for key,val in config[sect].items():
            if key in int_vals:
                val = int(val)
            if key in float_vals:
                val = float(val)
            if key in str_list_vals:
                val = as_list(val)
            if key in int_list_vals:
                val = as_list(val,int)
                
            return_dict.update({key:val})
    
    #config.set("data","fname",str(fname))
    #with open(fname,"w") as f:
    #    config.write(f)
    
    return return_dict


def tokenizer(type="Okt"):
    """
    Should return a Callable as a tokenizer
    """
    if type=="Okt":    
        return Okt().pos
    else:
        raise ValueError("Only Okt supported now")

def build_vocab(config,tokenizer,is_word2vec=True,additional_corpus=None):
    print("\nBuild a vocabulary...\n")
    corpus = []
    freq_list = Counter()
    
    class_dir = join(config["base_path"],config["txt_path"])

    for cls in tqdm(os.listdir(class_dir)):
        for txt_dir in os.listdir(os.path.join(class_dir,cls)):

            with open(join(class_dir,cls,txt_dir)) as f:
                txt = f.read().strip()
            tokens = tokenizer(txt)
            tokens = list(map(lambda x:x[0],tokens)) # POS not required
            corpus.append(tokens)
    
    if is_word2vec :
        if additional_corpus:
            corpus.extend(additional_corpus)
        w2v_model=Word2Vec(sentences=corpus,size=config["embed_dim"],
            window=5, min_count=config["min_freq"], workers=4, sg=1)
        embedding_matrix = w2v_model.wv.vectors
        
        vocab = {}
        vocab["<pad>"]=0
        vocab["<unk>"]=1
        vocab["<eos>"]=2
        BASE = len(vocab)
        for i,word in enumerate(w2v_model.wv.index2word):
            vocab[word]=i+BASE
        
        special_vectors=np.random.uniform(-1,1,[BASE,config["embed_dim"]])
        
        embedding_matrix = np.vstack([special_vectors,embedding_matrix]).astype(np.float32)
        np.save(join(config["data_path"],config["pretrain_path"],config["embedding"]),embedding_matrix)
        
    else:
        _corpus = []
        for sent in corpus:
            _corpus.extend(sent)
        vocab_freq = Counter(_corpus).most_common()
        vocab={}
        vocab["<pad>"]=0
        vocab["<unk>"]=1
        vocab["<eos>"]=2
        for word,freq in vocab_freq:
            if freq >= config["min_freq"]:
                vocab[word]=len(vocab)
    
        print("\nVocab built, total {} vocabs\n".format(len(vocab)))
    
    config.update({"vocab_size":len(vocab)})
    return vocab
    
def txt2Token(config,tokenizer,vocab):
    
    print("\nTurn words into tokens...\n")
    return_dict={}
    txt_dir = join(config["base_path"],config["txt_path"])
    #txt_file_list = sorted(os.listdir(txt_dir))
    
    def tokenizer_wrapper(txt):
        """
        turn to ids, padding, apply stopWrods or stopPOS
        """
        max_len = config["max_len"]
        txt_re = re.sub(r"[^ㄱ-힣0-9]"," ",txt)
        tokens=tokenizer(txt_re)
        tokens = [token for token,pos in tokens \
            if pos not in config["stop_pos"] and token not in config["stop_words"]]
        tokens = list(map(lambda x:vocab.get(x,vocab["<unk>"]),tokens))    
        return tokens[:max_len]+[vocab["<pad>"]]*(max_len-len(tokens))
    
    for cls in tqdm(os.listdir(txt_dir)):
        for fname in os.listdir(join(txt_dir,cls)):
            txt_path = join(txt_dir,cls,fname)
            f = open(txt_path,"r")
            txt = f.read()
            f.close()
            tokens_padded = tokenizer_wrapper(txt)
            return_dict[fname] = tokens_padded
    print("Total {} text are converted to IDs".format(len(return_dict)))
    return return_dict        

def img2Raw(config):
    print("\nTurn images into bytes...\n")
    def image_to_raw(fdir):
        image=Image.open(fdir)
        image_raw=image.resize(config["img_size"][:2]).tobytes()
        image.close()
        return image_raw
    
    img_dir = join(config["base_path"],config["img_path"])
    return_dict = {}
    
    for cls in tqdm(os.listdir(img_dir)):
        for fname in os.listdir(join(img_dir,cls)):
            img_path = join(img_dir,cls,fname)
            img_bytes = image_to_raw(img_path)
            return_dict[fname] = img_bytes
            
    return return_dict   

def match_and_write(_id,example,name2img,name2token,tfwriter):
    
    img_fname = example["img_file"]
    txt_fname = example["txt_file"]
    _class = example["class"]

    def example_func(img,ids,label,_id):       
        example = tf.train.Example(features=tf.train.Features(feature={
            "img_raw":_bytes_feature(img),
            "text_ids":_int64_feature(ids),
            "label":_int64_feature([label]),
            "id":_int64_feature([_id])}))
        return example
    
    example = example_func(name2img[img_fname],name2token[txt_fname],_class,int(_id))
    
    tfwriter.write(example.SerializeToString())

def load_json(path):
    with open(path) as f:
        _dict= json.load(f)
    return _dict

def save_json(_dict,fdir):
    with open(fdir,"w") as f:
        json.dump(_dict,f)
"""
def prepare_dataset(config):
    # argparser로 is_word2vec, additional_corpus, pretrained,refit 이거 결정
    if not exists(join(config["base_path"],config["tfrecord_path"])):
        os.makedirs(join(config["base_path"],config["tfrecord_path"]))
    if not exists(config["pretrain_path"]):
        os.makedirs(config["pretrain_path"])

    train_dict = load_json(config["train_json"])
    eval_dict = load_json(config["eval_json"])
    test_dict = load_json(config["test_json"])
    _tokenizer = tokenizer("Okt") # argparser로 줄것 

    vocab=build_vocab(config,_tokenizer)
    
    with open("vocab.pkl","wb") as f:
        pickle.dump(vocab,f)

    name2token = txt2Token(config,_tokenizer,vocab)
    name2img = img2Raw(config)
    
    train_tfwriter=tf.io.TFRecordWriter(
        join(config["base_path"],config["tfrecord_path"],"train.record"))
    eval_tfwriter=tf.io.TFRecordWriter(
        join(config["base_path"],config["tfrecord_path"],"eval.record"))
    test_tfwriter=tf.io.TFRecordWriter(
        join(config["base_path"],config["tfrecord_path"],"test.record"))    
    
    for _id, example in train_dict.items():
        match_and_write(_id,example,name2img,name2token,train_tfwriter)
    train_tfwriter.close()
        
    for _id, example in eval_dict.items():
        match_and_write(_id,example,name2img,name2token,eval_tfwriter)
    eval_tfwriter.close()
        
    for _id, example in test_dict.items():
        match_and_write(_id,example,name2img,name2token,test_tfwriter)
    test_tfwriter.close()

def data_loader(config,usage,mode,shuffle,batch_size):
    #config = parse_config(args_path,"data")
    
    assert usage in ["train","eval","test"], "mode should be one of 'train','eval','test'"
    
    _parse_img_example,_parse_txt_example,_parse_single_example = config_wrapper_parse_funcs(config)

    if mode=="text" or mode=="txt":
        parse_func = _parse_txt_example
    elif mode=="image" or mode=="img":
        parse_func = _parse_img_example
    elif mode=="both" or mode==None:
        parse_func = _parse_single_example
    else:
        parse_func = _parse_single_example
    
    dataset = tf.data.TFRecordDataset(
        join(config["base_path"],config["tfrecord_path"],f"{usage}.record")
    )
    dataset = dataset.map(parse_func)
    if shuffle:
        dataset = dataset.shuffle(batch_size)
    dataset = dataset.batch(batch_size)
    return dataset
"""
