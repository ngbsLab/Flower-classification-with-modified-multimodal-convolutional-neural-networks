import tensorflow
from dataset.oxfordflower.tf_utils import *
from model.LearningType import Learning

class PretrainModel(tf.keras.Model):

    def __init__(self,model_name="MobileNetV2",is_pred=True):
        """
        Just a wrapper for a model in tf.keras.applications, adjust some layers for modified mcnn
        """
        super(PretrainModel,self).__init__()

        pool_size = pool_size_checker(model_name)
    
        self.is_pred = is_pred
        
        model_func = getattr(tf.keras.applications,model_name)
        
        self._model = model_func(include_top=False,input_shape=(224,224,3))
        self.pool2d = tf.keras.layers.MaxPool2D(pool_size)
        self.fcVec = tf.keras.layers.Dense(4096)
        
        if self.is_pred:
            self.fc = tf.keras.layers.Dense(102)
            
        self.flatten = tf.keras.layers.Flatten()
        
    def call(self,inp):
        out = self._model(inp)
        out = self.pool2d(out)
        out = self.fcVec(out)
        if self.is_pred:
            out = self.fc(out)
        out = self.flatten(out)
        return out

    def save_by_ckpt(self,save_path,mask_layers=[3]):
        _dict = {}
        offset = 0
        for i,layer in enumerate(self.layers):
            if i in mask_layers:
                offset += 1
                continue
            _dict[str(i-offset)] = layer
        path = tf.train.Checkpoint(**_dict).save(save_path)
        return path,_dict
    
    def load_by_ckpt(self,saved_path,is_assert=False):
        _dict_re={}
        for i,layer in enumerate(self.layers):
            _dict_re[str(i)] = layer
        restore_status = tf.train.Checkpoint(**_dict_re).restore(saved_path)
        if is_assert:
            restore_status.assert_consumed()

class conv_block(layers.Layer):
    """
    Conv, (optional)BatchNormal, Dropout, Maxpool2d
    """
    def __init__(self,num_filters, kernel_size, rate, max_seq=None, is_bn=True):
        super(conv_block,self).__init__()
        self.conv = layers.Conv2D(num_filters, kernel_size=kernel_size, padding='valid', kernel_initializer='he_normal', activation='relu')
        self.bn =  layers.BatchNormalization()
        self.drop_rate= rate
        _poolsize =  [max_seq - kernel_size[0] + 1,1] if max_seq is not None else [2,1]
        _stride= [1,1] if max_seq is not None else None
        self.pool2d = layers.MaxPool2D(pool_size=_poolsize,strides=_stride,padding="valid")
        self.is_bn = is_bn
        
    def call(self,inp,train=None):
        out=self.conv(inp)
        if self.is_bn :
            out= self.bn(out,train)
        out = tf.nn.dropout(out,self.drop_rate)
        return self.pool2d(out)

class Modified_m_CNN(tf.keras.Model):
    
    def __init__(self,config,save_path,model_name,preset_trainable):
        super(Modified_m_CNN,self).__init__()
        self.config = config

        LAMBDA=self.config["lambda"]
        DROP_OUT=self.config["drop_out"]
        DROP_OUT2=self.config["drop_out2"]
        dropouts = [DROP_OUT,DROP_OUT2]
        kernel_sizes=self.config["filter_sizes"]
        num_filters = self.config["num_filters"]
        
        if "vocab_size" not in config:
            with open(config["vocab"],"rb") as f:
                self.vocab_size = len(pickle.load(f))
        else:
            self.vocab_size = config["vocab_size"]
        
        self.preset_trainable = preset_trainable
        self.preset_model = self.preset_model_loader(save_path,model_name,is_trainable=preset_trainable)
        self.bn = layers.BatchNormalization()
        self.conv_blockx = conv_block(256,(14,1),dropouts[1],None,is_bn=False)
        self.conv_block0 = conv_block(num_filters,(kernel_sizes[0], config["embed_dim"]),dropouts[1],config["max_len"])
        self.conv_block1 = conv_block(num_filters,(kernel_sizes[1], config["embed_dim"]),dropouts[1],config["max_len"])
        self.conv_block2 = conv_block(num_filters,(kernel_sizes[2], config["embed_dim"]),dropouts[1],config["max_len"])
        self.conv_block3 = conv_block(512,(5,1),dropouts[0],None)
        self.fc = layers.Dense(config["num_class"],kernel_initializer='he_normal')
        
        embedding_path = os.path.join(config["pretrain_path"] ,config["embedding"])
        if os.path.exists(embedding_path):
            print("\nPretrained skipgram,Embedding matrix loaded, from {}\n".format(embedding_path))
            embedding_array = np.load(embedding_path).astype(np.float32)
            initializer = tf.keras.initializers.Constant(embedding_array)
        else:
            initializer = "uniform"
        self.embed = layers.Embedding(self.vocab_size,config["embed_dim"],
                            input_length=config["max_len"],embeddings_initializer=initializer)
                            
        self.dropout_layer = layers.Dropout(dropouts[0])
        
        self.flatten = tf.keras.layers.Flatten()
        
        
    def call(self,inp,train=None):
        img, txt = inp["img"],inp["txt"] #inp["image"],inp["text"]
        img_vec = self.preset_model(img)
        # Image through vgg
        img_vec = tf.reshape(img_vec,[-1,16,1,256])
        img_vec = self.bn(img_vec,train)
        conv_x = self.conv_blockx(img_vec)
        
        # Text through embedding
        txt_embedded = self.embed(txt)
        txt_embeddded =  tf.expand_dims(txt_embedded,3)
        #tf.reshape(txt_embedded,[-1,config["max_len"],config["embed_dim"],1])
        conv_0 = self.conv_block0(txt_embeddded)
        conv_1 = self.conv_block1(txt_embeddded)
        conv_2 = self.conv_block2(txt_embeddded)
        
        concat1 = tf.concat([conv_0,conv_x],axis=1)
        concat2 = tf.concat([conv_1,conv_x],axis=1)
        concat3 = tf.concat([conv_2,conv_x],axis=1)
        
        concat_total = tf.concat([concat1,concat2,concat3],axis=1)
        conv_total = self.conv_block3(concat_total)
        #conv_total = tf.squeeze(conv_total)
        conv_total = self.dropout_layer(conv_total)
        
        output = self.fc(conv_total)
        output = self.flatten(output)
        return output
    
    @staticmethod
    def preset_model_loader(save_path=None,model_name=None,is_trainable=False):
        assert model_name is not None, "Please specifiy image encoder model, one of tf.keras.applications"
        if save_path is None:
            print("\nGet preset model, {}, trained with ImageNet\n".format(model_name))
            _model = PretrainModel(model_name,is_pred=False)
        else:
            _model = PretrainModel(model_name,is_pred=False)
            _model.load_by_ckpt(save_path)
            print("\nGet preset model, {}, trained with Target Data, from {}\n".format(model_name,save_path))
            
        _model.trainable=is_trainable
        
        return _model

    
class Hetero(Learning):
    def __init__(self,args,database,network_cls,encoder_name):
        """
        Take keras.Model and Database, and pretrain an Image encoder part of the Model
        
        Argument
            args : ArgumentParse (encoder_name , batch_size, epochs, etc)
            database : one of subclass of Database
            network_cls : tf.keras.Model(mcnn or VQA)
            encoder_name : (string )image classfication model, one of tf.keras.applications
        """
        
        self.config = parse_config()
        self.args = args
        self.database = database
        self.network_cls = network_cls
        
        # super(Hetero,self).__init__(args) # 아직 구현하지 못했는데, 여러상황들 고려해서 다시 구성할 예정입니다.
        
    def get_train_dataset(self):
        return self.database.data_loader("train","both",args.batch_size,args.batch_size)

    def get_val_dataset(self):
        return self.database.data_loader("eval","both",args.batch_size,args.batch_size)

    def get_test_dataset(self):
        return self.database.data_loader("test","both",args.batch_size,args.batch_size)
        
    def train(self):
        config = self.config    
        args = self.args
        
        if args.binary:
            loss_func = losses.BinaryCrossentropy(from_logits=True)
        else:
            loss_func = losses.SparseCategoricalCrossentropy(from_logits=True)
    
        if args.pretrain:
            path,_dict = self.pretrain_img_model(config,model_name=args.model_name,batch_size=args.batch_size2,epochs=args.epochs2,
                                        loss=loss_func,optimizer=args.optimizer,metrics=["accuracy"])
        else:
            path = None

        trn_dataset = self.get_train_dataset()
        eval_dataset = self.get_val_dataset()
        test_dataset = self.get_test_dataset()
    
        self.model = self.network_cls(config,path,args.model_name,preset_trainable=args.fineTune)
    
        if not exists(join(config["log_path"],config["csv_path"])):
            makedirs(join(config["log_path"],config["csv_path"]))
        ckpt_cb = callbacks.ModelCheckpoint(join(config["log_path"],config["ckpt_path"],f"{args.model_name}.ckpt"))
        csv_log_cb = callbacks.CSVLogger(join(config["log_path"],config["csv_path"],"log.csv"))
    
        self.model.compile(loss =loss_func,optimizer=args.optimizer,metrics=["accuracy"])
        self.model.fit(trn_dataset,epochs=args.epochs,callbacks=[ckpt_cb,csv_log_cb],validation_data = eval_dataset)
        #self.model.evaluate(test_dataset)
        
    def predict(self,inp):
        return self.model(inp)
    
    def evaluate(self):
        pass
    
    def pretrain_img_model(self,config,model_name="MobileNetV2",usage="train",batch_size=64,epochs=30,**kwargs):
    
        tf_path = os.path.join(config["base_path"],"tfrecord",usage+".record")
    
        _parse_img_example,_,_ = config_wrapper_parse_funcs(config)

        dataset = tf.data.TFRecordDataset(tf_path).map(_parse_img_example)
        dataset = dataset.shuffle(batch_size).batch(batch_size)
    
        preset_model = PretrainModel(model_name=model_name,is_pred=True)

        preset_model.compile(**kwargs)
    
        preset_model.fit(dataset,epochs=epochs)
      
        path,_dict = preset_model.save_by_ckpt(
            os.path.join(config["pretrain_path"],model_name,"ck.ckpt"),
            mask_layers=[3])
            
        return path,_dict

    def get_config_info(self):
        whole_dict = {}
        whole_dict.update(self.args)
        whole_dict.update(self.config)
        return whole_dict
