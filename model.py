from requirements import *


def LSTM_model(vocab_size, max_length):
  # features from the CNN model compressed from 2048 to 256 nodes
   inputs1 = Input(shape=(299,299,3))
   model = Xception( include_top=False, pooling='avg')
   model.trainable = False
   img_features = model(inputs1)
   fe1 = Dropout(0.5)(img_features)
   fe2 = Dense(256, activation='relu')(fe1)
  # LSTM sequence model
   inputs2 = Input(shape=(max_length,))
   se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
   se2 = Dropout(0.5)(se1)
   se3 = LSTM(256)(se2)
  # Merging both models
   decoder1 = add([fe2, se3])
   decoder2 = Dense(256, activation='relu')(decoder1)
   outputs = Dense(vocab_size, activation='softmax')(decoder2)
  # merge it [image, seq] [word]
   model = Model(inputs=[inputs1, inputs2], outputs=outputs)
   model.compile(loss='categorical_crossentropy', optimizer='adam')
  # summarize model
   print(model.summary())
   return model