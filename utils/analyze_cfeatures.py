from pickle import load
import numpy as np
import matplotlib.pyplot as plt
from safe_cfeatures import load_obj

def analyze_cfeatures():
    result_dict = load_obj()
    all_cfeatures = []
    for prompt in result_dict:
        prompt_dict = result_dict[prompt]
        for num_event, event in enumerate(prompt_dict):
            all_cfeatures.append(event['cfeatures'])

    #print(all_cfeatures)
    np_c_features = np.array(all_cfeatures)
    plt.plot(np_c_features[:,0])
    plt.show()
if __name__ == '__main__':
    analyze_cfeatures()