from json.tool import main
import torch
from pickle import load
import numpy as np
import matplotlib.pyplot as plt
from safe_cfeatures import load_obj
from sklearn.decomposition import PCA
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 20}

plt.rc('font', **font)

def normalize(inpt):
    vec_min = np.min(inpt, axis=0)
    vec_max = np.max(inpt, axis=0)
    amp_2 = 2*(inpt-vec_min) / ((vec_max - vec_min) + 1e-3)
    amp_2 = amp_2 - 1
    return amp_2, vec_min, vec_max

def apply_norm(inpt, vec_min, vec_max):
    amp_2 = 2*(inpt-vec_min) / ((vec_max - vec_min) + 1e-3)
    amp_2 = amp_2 - 1
    return amp_2

def get_pca(np_array, norms = None, choose_indices = None, pca = None):
    #np_array = np.array(list)
    if choose_indices is not None:
        np_array = np_array[:, choose_indices]
    
    if norms is None:
        norm_array, vec_min, vec_max = normalize(np_array)
    else:
        norm_array = apply_norm(np_array, norms[0], norms[1])
    
    if pca is not None:
        pca_man = pca.transform(norm_array)
        return pca_man
    else:
        pca = PCA(n_components = 2)
        pca_result = pca.fit_transform(norm_array)
        return pca, pca_result, [vec_min, vec_max]

def plot_n_scatter(datas, colors = ['r', 'g'], names = ['a', 'b'], x_label = 'Principal Component 1', y_label = 'Principal Component 2', title = '2 component PCA', annotate = None):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel(x_label) 
    ax.set_ylabel(y_label) 
    ax.set_title(title) 
    for data, color in zip(datas, colors):
        ax.scatter(data[:,0], data[:,1] , c = color, s = 50)
    ax.legend(names)
    ax.grid()
    if annotate is not None:
        for text, x, y in zip(*annotate):
            ax.annotate(text, (x, y))
    plt.show()

def analyze():
    path = '/home/hendrik/Documents/master_project/LokalData/presi/lang'
    language = torch.load(path)
    language = language.cpu().numpy()
    print(language.shape)
    cfeat_pca, cfeat_pca_result, norm_pca = get_pca(language[:100])
    ora= get_pca(language[100:], pca = cfeat_pca, norms=norm_pca)
    plot_n_scatter([cfeat_pca_result, ora], colors=['g', 'r'], names=['training', 'ood'], title='language')

analyze()