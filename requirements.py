import numpy as np
from PIL import Image
import argparse
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import json
import itertools
import os
import string
import pickle
from pickle import dump
from pickle import load
import tensorflow as tf
from tensorflow.keras.applications.xception import Xception #to get pre-trained model Xception
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer #for text tokenization
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import add
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense#tensorflow.keras to build our CNN and LSTM
from tensorflow.keras.layers import LSTM, Embedding, Dropout
from tqdm import tqdm #to check loop progress
tqdm().pandas()

        