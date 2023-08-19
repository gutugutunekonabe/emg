import pandas as pd
import numpy as np

LOAD_DATA_DIR_PATH = "C:\\Users\\Alienware\\Desktop\\csvraw\\try\\nm0.csv"

reader = pd.read_csv(LOAD_DATA_DIR_PATH, chunksize=999)


for X in range(8000):

  reader.get_chunk(999).to_csv(str(X)+'.csv', header=True, index=False)

  
  

 