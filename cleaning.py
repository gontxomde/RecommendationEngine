import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import math, nltk, warnings
from nltk.corpus import wordnet
from sklearn import linear_model
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import fuzz
from wordcloud import WordCloud, STOPWORDS

from common import load_tmdb_credits
from common import load_tmdb_movies
from common import convert_to_original_format

class Cleaner():
    def __init__ (self,df):
        self.df = df.copy(deep = True)
    
    def clean_df(self):

        return self.df
