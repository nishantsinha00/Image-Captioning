from requirements import *

# Load the document file into memory
def load_fp(filename):
  # Open file to read
   file = open(filename, 'r')
   text = file.read()
   file.close()
   return text

# get all images with their captions
def img_capt(filename):
   file = load_fp(filename)
   captions = file.splitlines()
   descriptions ={}
   for caption in captions[:-1]:
          img, caption = caption.split('\t')
          if img[:-2] not in descriptions:
              descriptions[img[:-2]] = [ caption ]
          else:
              descriptions[img[:-2]].append(caption)
   return descriptions

#Data cleaning function will convert all upper case alphabets to lowercase, removing punctuations and words containing numbers
def txt_clean(captions):
   table = str.maketrans('','',string.punctuation)
   for img,caps in captions.items():
       for i,img_caption in enumerate(caps):
              img_caption.replace("-"," ")
              descp = img_caption.split()
             #uppercase to lowercase
              descp = [wrd.lower() for wrd in descp]
             #remove punctuation from each token
              descp = [wrd.translate(table) for wrd in descp]
             #remove hanging 's and a
              descp = [wrd for wrd in descp if(len(wrd)>1)]
             #remove words containing numbers with them
              descp = [wrd for wrd in descp if(wrd.isalpha())]
             #converting back to string
              img_caption = ' '.join(descp)
              captions[img][i]= img_caption
   return captions

def txt_vocab(descriptions):
  # To build vocab of all unique words
   vocab = set()
   for key in descriptions.keys():
          [vocab.update(d.split()) for d in descriptions[key]]
   return vocab

#To save all descriptions in one file
def save_descriptions(descriptions, filename):
    with open(filename, "w") as outfile:
        json.dump(descriptions, outfile, indent=4)
# Set these path according to project folder in you system, like i create a folder with my name shikha inside D-drive

if __name__=='__main__':
    dataset_folder = "Data"

    #to prepare our text data
    filename = dataset_folder + "/" + "Flickr8k.token.txt"
    #loading the file that contains all data
    #map them into descriptions dictionary 
    descriptions = img_capt(filename)
    print("Length of descriptions =" ,len(descriptions))
    #cleaning the descriptions
    clean_descriptions = txt_clean(descriptions)
    #to build vocabulary
    vocabulary = txt_vocab(clean_descriptions)
    print("Length of vocabulary = ", len(vocabulary))
    #saving all descriptions in one file
    save_descriptions(clean_descriptions, "descriptions.json")