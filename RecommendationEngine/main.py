
from recommendation import Recommendator
import os
import yaml
import pandas as pd

def main():
    RECOM = Recommendator("config_partial.yml")
    
    RECOM.user_recommendation()

if __name__ == "__main__":
    main()