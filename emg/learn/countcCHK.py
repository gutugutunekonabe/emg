import csv
import collections
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

filepath = 'nyanko/test.csv' 

count = {}
with open(filepath) as f:
    reader = csv.reader(f)
    for row in reader:
        onigiri = row[0]
        count.setdefault(onigiri, 0)
        count[onigiri] +=1
for key, value in count.items():
   
    print('{}: {}'.format(key, value))     
