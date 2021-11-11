from pickle import load
import numpy as np
import matplotlib.pyplot as plt
from safe_cfeatures import load_obj
from sklearn.decomposition import PCA

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 20}

plt.rc('font', **font)

def analyze_obj_attention():
    result_dict = load_obj(name='dict_of_features_complete')
    attention = extract_feature(result_dict, 'attn', event_condition('success'))

    man_result_dict = load_obj('dict_of_features_manual')
    attn_man = extract_feature(man_result_dict, 'attn', event_condition())
    prompts_man = extract_feature(man_result_dict, 'prompt', event_condition())[1:]

    max_attention_success = attention[:-1,0].max(axis=1)
    max_attention_fail = attn_man[1:,0].max(axis = 1)
    scatter_attn = np.array([np.arange(len(max_attention_success)),max_attention_success]).T
    scatter_att_fail = np.array([np.arange(len(max_attention_fail)),max_attention_fail]).T
    
    annotation = (prompts_man,
    [scatter_att_fail[0,0]],
    [scatter_att_fail[0,1]]
    )


    plot_n_scatter([scatter_attn, scatter_att_fail], colors=['g', 'r'], names=['success', 'fail'], title='Object attention', annotate=annotation)

def analyze_cfeatures():
    result_dict = load_obj(name='dict_of_features_complete')
    all_cfeatures = extract_feature(result_dict, 'cfeatures', event_condition('success'))[:-1]
    #success = extract_feature(result_dict, 'success', event_condition('success'))
    prompts = extract_feature(result_dict, 'prompt', event_condition('success'))
    #attention = extract_feature(result_dict, 'attn', event_condition('success'))

    man_result_dict = load_obj('dict_of_features_manual')
    c_features_man = extract_feature(man_result_dict, 'cfeatures', event_condition())
    prompts_man = extract_feature(man_result_dict, 'prompt', event_condition())
    #one example prompt
    prompts_man = np.append(prompts_man, prompts[0])
    
    #last one is a spilling, error by intendation...
    #extract class and language
    class_and_language = [0]
    for i in range(5, 37):
        class_and_language.append(i)

    #print(plt.plot(attention[:,0].max(axis=1)))
    #plt.show()

    cfeat_pca, cfeat_pca_result, norm_pca = get_pca(all_cfeatures)
    cfeat_man_pca = get_pca(c_features_man, norms=norm_pca, pca = cfeat_pca)
    #plot_n_scatter([attn_pca_result, attn_man_pca], colors=['g', 'r'], names=['success', 'failure, ambiguous'], title='Object attention')
    annotation = (prompts_man,
    [cfeat_man_pca[0,0], cfeat_man_pca[1,0], cfeat_pca_result[0,0]],
    [cfeat_man_pca[0,1], cfeat_man_pca[1,1], cfeat_pca_result[0,1]]
    )
    plot_n_scatter([cfeat_pca_result, cfeat_man_pca], colors=['g', 'r'], names=['success', 'fail'], title='cfeature', annotate=annotation)


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


def extract_feature(dictionary, feature, cond):
    all_features = []
    for prompt in dictionary:
        prompt_dict = dictionary[prompt]
        for num_event, event in enumerate(prompt_dict):
            if cond(event):
                all_features.append(event[feature])
    return np.array(all_features)

def event_condition(phrase = None):
    if phrase is None:
        return lambda event:True
    else:
        return lambda event: phrase in event

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



if __name__ == '__main__':
    #analyze_cfeatures()
    analyze_obj_attention()
