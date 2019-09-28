# *-* coding:utf-8 *-*
import random
import os
import sys

# 我只有A卡，所以用了pml后端
import plaidml.keras
plaidml.keras.install_backend()

import keras
import numpy as np
from keras.models import Sequential
from keras.callbacks import LambdaCallback
from keras.models import Input, Model, load_model
from keras.layers import LSTM, Dropout, Dense, Flatten, Bidirectional, Embedding, GRU, Conv1D, MaxPool1D
from keras.optimizers import Adam

class midiModel(object):
    def __init__(self,path):
        self.model = None
        self.model_path = path
        if os.path.exists(path):
            print("load model")
            self.model = load_model(path)
            self.model.summary()
        else:
            print("build model")
            self.build_model()
    
    def build_model(self):
        
        self.model = Sequential()
        self.model.add(GRU(128 , return_sequences=True , input_shape=(32,128)))
        self.model.add(Dropout(0.6))
        self.model.add(GRU(128 , return_sequences=True))
        self.model.add(Dense(129, activation='softmax'))
        
        optimizer = Adam(lr=0.001)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        self.model.summary()
    
    def load_file(self,path):
        notes = []
        num = 0
        with open(path, 'r', encoding='utf-8') as f:
            print("open file:"+path)
            for line in f:
                out_arr = line.split("=",1)
                if len(out_arr)>=1 :
                    if len(out_arr)==2 :
                        out = int(out_arr[1])
                    else:
                        out = -1
                    
                    inp_arr = out_arr[0].split(",")
                    
                    inp = []
                    if len(inp_arr)>0 :
                        for inp_str in inp_arr:
                            inp.append(inp_str)
                    
                    notes.append([inp,out])
                
            return num,notes
    
    def train(self,path):
        # 加载文件
        data_len , notes = self.load_file(path)
        
        count = 0;
        x = np.zeros(
            shape=(32 , 128)
        )
        y = np.zeros(
            shape=(32 , 129)
        )
        print("start")
        for note in notes:
            
            note_inp = note[0]
            note_out = note[1]
            if note_out>=0 and note_out<128:
                y[count][note_out] = 1
            else:
                y[count][128]=1;
            for ps in note_inp:
                posi = int(ps)
                if posi>=0 and posi<128:
                    x[count][posi] = 1
            
            if count >=31 :
                count = 0
                #print ("===========================================================")
                #print (x)
                #print (y)
                #print ("===========================================================")
                # 开始训练
                self.model.fit(
                    [[x]],[[y]], # 我也不知道为什么要这样写。这是试出来的
                    verbose=True,
                    steps_per_epoch=32,
                    epochs=1,
                    callbacks=[
                        keras.callbacks.ModelCheckpoint(self.model_path, save_weights_only=False)
                    ]
                )
                
                x = np.zeros(
                    shape=(32 , 128)
                )
                y = np.zeros(
                    shape=(32 , 129)
                )
            else:
                count = count+1
    
    def predict(self,notes):
        x = np.zeros(
            shape=(32 , 128)
        )
        count = 0
        for ser in notes:
            for posi in ser:
                #print(count , posi)
                x[count][int(posi)] = 1
            count = count+1
            if count>=32 :
                break
        res_vec = self.model.predict([[x]])[0]
        res=[]
        for it in res_vec:
            mp = np.argmax(it)
            if mp>=128 or mp<0 :
                mp = -1
            res.append(mp)
            
        return res
        
if __name__ == '__main__' :
    model = midiModel("model.h5")
    model.train("data.txt")
    print("test:")
    print(model.predict([[1,2,3],[2,3,4]]))
    
