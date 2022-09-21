from requirements import *

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help="Image Path")
args = vars(ap.parse_args())
img_path = args['image']


def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_desc(model, tokenizer, photo, max_length):
   in_text = 'start'
   for i in range(max_length):
       sequence = tokenizer.texts_to_sequences([in_text])[0]
       sequence = pad_sequences([sequence], maxlen=max_length)
       pred = model.predict([photo,sequence], verbose=0)
       pred = np.argmax(pred)
       word = word_for_id(pred, tokenizer)
       if word is None :
           break
       in_text += ' ' + word
       if word == 'end':
           break
       
   return in_text

max_length = 32
tokenizer = load(open("tokenizer.p","rb"))
model = load_model('models/model_9.h5')
xception_model = Xception(include_top=False, pooling="avg")

img = cv2.imread(img_path)
img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (299,299)) 
img = img/127.5
img = img - 1.0
photo = np.expand_dims(img, axis=0)

description = generate_desc(model, tokenizer, photo, max_length)
print("nn")
print(description)
plt.imshow(img)