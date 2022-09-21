from requirements import *
from utils import *

def extract_features(directory):
       model = Xception( include_top=False, pooling='avg' )
       features = {}
       for pic in tqdm(os.listdir(directory)):
           file = directory + "/" + pic
           image = Image.open(file)
           image = image.resize((299,299))
           image = np.expand_dims(image, axis=0)
          #image = preprocess_input(image)
           image = image/127.5
           image = image - 1.0
           feature = model.predict(image)
           features[pic] = feature
       return features

def load_photos(filename):
   file = load_fp(filename)
   photos = file.splitlines()
   return photos

def load_clean_descriptions(filename, photos):
  #loading clean_descriptions
   descriptions = {}
   with open(filename, 'r') as f:
       clean_descriptions = json.load(f)

   for image in photos:
       if image not in descriptions:
           descriptions[image] = clean_descriptions[image]

   return descriptions


def load_features(photos):
  #loading all features
   all_features = load(open("features.pb","rb"))
  #selecting only needed features
   features = {k:all_features[k] for k in photos}
   return features

#convert dictionary to clear list of descriptions
def dict_to_list(descriptions):
   all_desc = list(itertools.chain(*[*descriptions.values()]))
   return all_desc


def create_tokenizer(descriptions):
   desc_list = dict_to_list(descriptions)
   tokenizer = Tokenizer()
   tokenizer.fit_on_texts(desc_list)
   return tokenizer


def max_length(descriptions):
   desc_list = dict_to_list(descriptions)
   return max(len(d.split()) for d in desc_list)



if __name__=='__main__':
    dataset_path = 'Data'
    datset_text = dataset_path + "/" + "Flickr_8k.trainImages.txt"
    dataset_images = dataset_path + "/" "Flicker8k_Dataset"

    with open("descriptions.json",'r') as f:
        descriptions = json.load(f)
        
    features = extract_features(dataset_images)
    dump(features, open("features.pb","wb"))
    
    
    tokenizer = create_tokenizer(descriptions)
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    vocab_size = len(tokenizer.word_index) + 1
    max_length = max_length(descriptions)
    
    config_dict = {
        "vocab_size": vocab_size,
        "max_length": max_length
    }
    
    with open('model_config.json', 'w') as f:
        json.dump(config_dict, f, indent=4)