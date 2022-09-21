from requirements import *
from model import LSTM_model
from tokenizer import *
from data import Datagen

def data_generator(descriptions, features, tokenizer, max_length, vocab_size):
    while 1:
        for key, description_list in descriptions.items():
          #retrieve photo features
           feature = features[key][0]
           inp_image, inp_seq, op_word = create_sequences(tokenizer, max_length, description_list, feature, vocab_size)
           yield [[inp_image, inp_seq], op_word]
           
def create_sequences(tokenizer, max_length, desc_list, feature, vocab_size):
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
           x_1.append(feature)
           x_2.append(in_seq)
           y.append(out_seq)
   return np.array(x_1), np.array(x_2), np.array(y)

if __name__=='__main__':

    image_path = 'Data\Flicker8k_Dataset'
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    with open('descriptions.json', 'r') as handle:
        descriptions = json.load(handle)
    with open('model_config.json', 'r') as handle:
        model_config = json.load(handle)
    train_images = load_photos('Data\Flickr_8k.trainImages.txt')
    val_images = load_photos('Data\Flickr_8k.testImages.txt')
    
    
    
    training_generator = Datagen(train_images, image_path, descriptions, tokenizer, model_config['vocab_size'], model_config['max_length'])
    validation_generator = Datagen(val_images, image_path, descriptions, tokenizer, model_config['vocab_size'], model_config['max_length'])
    model = LSTM_model( model_config['vocab_size'], model_config['max_length'])
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    epochs = 10
    model.fit(training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    epochs = epochs)
    model.save("models/model_.h5")
    