from preprocessing import Preprocesser
from imputation import Imputer
from recommendation import Recommendator
import os
import yaml
import pandas as pd

def main():
    with open('config.yml') as f:
        
        config = yaml.load(f)
        #print(config)
    complete_execution = config['complete_execution']

    if complete_execution:
        PP = Preprocesser(movies_path = config['movies'], credits_path= config['credits'])
        df = PP.preprocess()

        IM = Imputer(df)

        df = IM.impute()
        df.to_pickle("dfPickle.pkl")
    else:
        df = pd.load_("dfPickle.pkl")
    
    REC = Recommendator(df)
    string = input("Introduce el nombre de la película: ")
    REC.predict_from_string(string)


if __name__ == "__main__":
    main()