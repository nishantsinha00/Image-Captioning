from ctypes.wintypes import RGB
from requirements import *
import json
import math
import cv2

class Datagen(tf.keras.utils.Sequence):

    def __init__(self, image_list, image_path, descriptions, tokenizer, vocab_size, max_length):
        self.image_list = image_list
        self.image_path = image_path
        self.descriptions =  descriptions
        self.vocab_size = vocab_size
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.image_list)
    
    def create_sequences(self, tokenizer, max_length, desc_list, image, vocab_size):
        x_1, x_2, y = list(), list(), list()

        for desc in desc_list:
           # encode the sequence
            seq = tokenizer.texts_to_sequences([desc])[0]
            for i in range(1, len(seq)):
               # divide into input and output pair
                in_seq, out_seq = seq[:i], seq[i]
               # pad input sequence
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
               # encode output sequence
                out_seq = to_categorical([out_seq], num_classes= vocab_size)[0]
               # store
                x_1.append(image)
                x_2.append(in_seq)
                y.append(out_seq)
        return np.array(x_1), np.array(x_2), np.array(y)
     
    def __getitem__(self, idx):

        image = self.image_list[idx]
        description_list = self.descriptions[image]
        
        image_p = os.path.join(self.image_path, image)
        img = cv2.imread(image_p)
        img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (299,299)) 
    
            
        img = img/127.5
        img = img - 1.0

        inp_image, inp_seq, op_word = self.create_sequences(self.tokenizer, self.max_length, description_list, img, self.vocab_size)
        return [inp_image, inp_seq], op_word
        