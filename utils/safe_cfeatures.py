import numpy as np
import pickle
import os

def save_dict_of_features(dict_of_features, language = None, name = 'dict_of_features', path = 'utils/collected_data/', override = False):

    if not override:
        try:
            dictionary = load_obj(name, path)
        except:
            dictionary = {}
        if language in dictionary:
            dictionary[language].append(dict_of_features)
        else:
                dictionary[language] = [dict_of_features]
    else:
        dictionary = dict_of_features
    
    with open(path + name + '.pkl', 'wb') as f:
        pickle.dump(dictionary, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name = 'dict_of_features', path = 'utils/collected_data/'):
    path_to_file = path + name + '.pkl'
    with open(path_to_file, 'rb') as f:
        return pickle.load(f)

def delete_dict(name = 'dict_of_features', path = 'utils/collected_data/'):
    path_to_file = path + name + '.pkl'
    if os.path.exists(path_to_file):
        os.remove(path_to_file)

if __name__ == '__main__':
    dict_of_features = {
        'cfeatures' : np.array([1,2,3]),
        'attn' : np.array([1.,3])
    }
    save_dict_of_features(dict_of_features=dict_of_features, language='asd1')
    save_dict_of_features(dict_of_features=dict_of_features, language='asd2')
    save_dict_of_features(dict_of_features=dict_of_features, language='asd1')
    
    loaded_dict = load_obj('dict_of_features')
    #print(loaded_dict)
    dict_of_prompt = loaded_dict['asd1'][-1]
    #print(dict_of_prompt)
    dict_of_prompt['success'] = True
    print(loaded_dict)
    save_dict_of_features(loaded_dict, override=True)
    nd = load_obj('dict_of_features')
    #print(nd)
    #delete_dict('dict_of_features')