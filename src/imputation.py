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
from common import get_synonyms, keywords_inventory


class Imputer():
    def __init__ (self,df):
        self.df = df.copy(deep = True)
        self.keywords_inventory = keywords_inventory
    
    def impute(self):

        self.fill_year()
        keywords, _, _ = \
        keywords_inventory(self.df, column = 'plot_keywords')
        # Add keywords from the title
        icount = 0
        for index, row in self.df[self.df['plot_keywords'].isnull()].iterrows():
            icount += 1
            word_list = row['movie_title'].strip().split()
            new_keyword = []
            for s in word_list:
                lemma = get_synonyms(s)
                for t in list(lemma):
                    if t in keywords: 
                        new_keyword.append(t)                
            if new_keyword:
                self.df.at[index, 'plot_keywords'] = '|'.join(new_keyword)

        # Complete gross with num_voted_users
        self.df = self.variable_linreg_imputation(self.df, 'gross', 'num_voted_users')
        self.df.reset_index(inplace = True, drop = True)

        
        return self.df

    def fill_year(self):
        """Completa la columna faltante del año teniendo en cuenta la media
        de los periodos de actividad de los actores y el director.
        """

        col = ['director_name', 'actor_1_name', 'actor_2_name', 'actor_3_name']
        usual_year = [0 for _ in range(4)]
        var        = [0 for _ in range(4)]
        #_____________________________________________________________
        # Año medio de actividad para los actores y el director
        for i in range(len(col)):
            usual_year[i] = self.df.groupby(col[i])['title_year'].mean()
        #_____________________________________________
        # Diccionario que recoja esta información
        actor_year = dict()
        for i in range(4):
            for s in usual_year[i].index:
                if s in actor_year.keys():
                    if pd.notnull(usual_year[i][s]) and pd.notnull(actor_year[s]):
                        actor_year[s] = (actor_year[s] + usual_year[i][s])/2
                    elif pd.isnull(actor_year[s]):
                        actor_year[s] = usual_year[i][s]
                else:
                    actor_year[s] = usual_year[i][s]
            
        #______________________________________
        # Identificación de los años faltantes
        missing_year_info = self.df[self.df['title_year'].isnull()]
        #___________________________
        # Completado de los valores faltantes
        icount_replaced = 0
        for index, _ in missing_year_info.iterrows():
            value = [ np.NaN for _ in range(4)]
            icount = 0 ; sum_year = 0
            for i in range(4):            
                var[i] = self.df.loc[index][col[i]]
                if pd.notnull(var[i]): value[i] = actor_year[var[i]]
                if pd.notnull(value[i]): icount += 1 ; sum_year += actor_year[var[i]]
            if icount != 0: sum_year = sum_year / icount 

            if int(sum_year) > 0:
                icount_replaced += 1
                self.df.at[index, 'title_year'] = int(sum_year)

    def variable_linreg_imputation(self, df, col_to_predict, ref_col):
        """Completa los valores de la variable col_to_predict haciendo una regresión
        lineal en la que la variable predictora es ref_col.
        
        Args:
            df (pd.DataFrame): DataFrame de películas
            col_to_predict (str): Variable a predecir
            ref_col (str): Variable con la que predecir
        
        Returns:
            pd.DataFrame: DataFrame de películas completado
        """

        regr = linear_model.LinearRegression()
        test = df[[col_to_predict,ref_col]].dropna(how='any', axis = 0)
        X = np.array(test[ref_col])
        Y = np.array(test[col_to_predict])
        X = X.reshape(len(X),1)
        Y = Y.reshape(len(Y),1)
        regr.fit(X, Y)
        
        test = df[df[col_to_predict].isnull() & df[ref_col].notnull()]
        for index, row in test.iterrows():
            value = float(regr.predict(row[ref_col]))
            df.at[index, col_to_predict] =  value
        return df