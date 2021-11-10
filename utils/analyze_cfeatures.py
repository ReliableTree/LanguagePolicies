from pickle import load
import numpy as np
import matplotlib.pyplot as plt
from safe_cfeatures import load_obj

def analyze_cfeatures():
    result_dict = load_obj()
    promt_dict = result_dict['lift the indigo stein ']
    for key in promt_dict[0]:
        print(key)
    print(promt_dict[0]['attn'][0])
    print(promt_dict[0]['features'])
    print(promt_dict[0]['cfeatures'])

if __name__ == '__main__':
    analyze_cfeatures()