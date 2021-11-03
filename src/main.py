from scipy.optimize import linear_sum_assignment
import numpy as np
from ops import *

def example():
    eng_vocab = ["outbreak",
         "legendary",
         "handball",
         "georgian",
         "copenhagen",
    ]
    eng_texts = [f"This is a photo of a " + desc for desc in eng_vocab]
    eng_params = {
        'clip_name': "ViT-B/32"
    }
    
    # italian
    ita_params = {
        'clip_name': 'ViT-B/32',
    }
    ita_vocab = [
        'epidemia',
        'leggendario',
        'pallamano',
        'georgiano',
        'copenhagen',
        ]
    ita_vocab.reverse()

    ita_texts = [f"Questa è una foto di a "+ desc for desc in ita_vocab]

    probs_EN = get_fingerprints(eng_texts, eng_params['clip_name'], is_clip=True, image_name='tiny', num_images=1)
    probs_FR = get_fingerprints(ita_texts, ita_params['clip_name'], is_clip=False, image_name='tiny', num_images=1)

    cost = -(probs_EN @ probs_FR.T)
    _, col_ind = linear_sum_assignment(cost)
    print(col_ind)
    from IPython import embed; embed()



def get_word_pairs(path, lower=True):
    """
    Return a list of (word1, word2, score) tuples from a word similarity file.
    """
    assert os.path.isfile(path) and type(lower) is bool
    word_pairs = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip()
            line = line.lower() if lower else line
            line = line.split()
            # ignore phrases, only consider words
            if len(line) != 3:
                assert len(line) > 3
                assert 'SEMEVAL17' in os.path.basename(path) or 'EN-IT_MWS353' in path
                continue
            word_pairs.append((line[0], line[1], float(line[2])))
    return word_pairs


def load_vocabs():
    with open('../en_it.txt') as f:
        lines = f.readlines()
    dict_xy = {}
    x_vocab = set()
    y_vocab = set()

    for i in range(len(lines)):
        l = lines[i]
        x, y = l.strip().lower().split(' ')
        x_vocab.add(x)
        y_vocab.add(y)
        if x in dict_xy:
            # pass
            dict_xy[x].append(y)
        else:
            dict_xy[x] = [y]
    return x_vocab, y_vocab, dict_xy

def run(eng_vocab, ita_vocab, dict_ei):
    # np.random.shuffle(eng_vocab)
    ita_vocab = list(ita_vocab)
    eng_vocab = list(eng_vocab)
    eng_texts = [f"This is a photo of a " + desc for desc in eng_vocab]
    ita_texts = [f"Questa è una foto di a " + desc for desc in ita_vocab]
    # ita_texts = [f"фото " + desc for desc in ita_vocab]
    clip_names = {
        'eng': "ViT-B/32",
        # 'ita': 'RN50x4'
        'rus': 'ViT-B/32'
    }
    probs_eng = get_fingerprints(eng_texts, clip_names['eng'], is_clip=True, image_name='tiny', num_images=5)
    probs_ita = get_fingerprints(ita_texts, clip_names['rus'], is_clip=False, image_name='tiny', num_images=5)

    y = probs_eng @ probs_ita.T 
    cost = -y
    row_ind, col_ind = linear_sum_assignment(cost)

    s = 0
    for i in range(len(eng_vocab)):
        # print(eng_vocab[i], ita_vocab[col_ind[i]])
        if ita_vocab[col_ind[i]] in dict_ei[eng_vocab[i]]:
            s+=1
    print('Accuracy: {:.4f}'.format(s/len(eng_vocab)))
    # from IPython import embed; embed()

    # np.save('../results/eng_tiny.npy', probs_eng)
    # np.save('../results/ita_tiny.npy', probs_ita)

    # images = np.load('../data/image_tiny_5_vit32.npy' , allow_pickle=True)
    # images = torch.Tensor(images)
    # language_model, image_model, _ = get_models('clip_italian')
    # text_features = precompute_text_features(language_model, ita_texts)
    # image_features = precompute_image_features(image_model, images)
    # CLIP Temperature scaler
    # probs_ita = cal_probs_from_features(image_features, text_features, logit_scale=1)


#     from sklearn.utils.extmath import randomized_svd
    # X = probs_eng @ probs_ita.T 
    # U, Sigma, VT = randomized_svd(X, n_components=1000, n_iter=5, random_state=42)
    # W = U @ VT
    # ita = W @ probs_ita
    # y = probs_eng @ ita.T
    # nxm
    # if False:
    
    # preds = np.argsort(y, axis=1)
    # predited_ita = [(eng_vocab[i], ita_vocab[preds[i]]) for i in range(len(eng_vocab))]
    # s = 0
    # for i in range(len(eng_vocab)):
    #     for k in range(5):
    #         if ita_vocab[preds[i][k]] in dict_ei[eng_vocab[i]]:
    #             s+=1
    #             break
    #         # print(ita_vocab[preds[i]], eng_vocab[i])
    # print('Accuracy: {:.4f}'.format(s/len(eng_vocab)))
    #     else:
    
    



def test():
    # italian
    from sentence_transformers import SentenceTransformer
    ita_vocab = [
        "la foto di una mela",
        "una foto di un computer",
        "la foto di un telefono",
        "una foto di un razzo",
        "una foto di una moto"
    ]
    ita_texts = [f"Questa è " + desc for desc in ita_vocab]

    images = np.load('../data/image_tiny_5_vit32.npy' , allow_pickle=True)
    images = torch.Tensor(images)
    # language_model, image_model = get_models('mClip')
    se_language_model = SentenceTransformer('clip-ViT-B-32-multilingual-v1')
    se_image_model = SentenceTransformer('clip-ViT-B-32')
    text_features = precompute_text_features(language_model, ita_texts)
    image_features = precompute_image_features(image_model, images)
    # CLIP Temperature scaler
    image_features = torch.Tensor(image_features).cuda()
    text_features = torch.Tensor(text_features).cuda()
    probs = cal_probs_from_features(image_features, text_features, logit_scale=1)



if __name__ == '__main__':
    # eng_vocab, ita_vocab, dict_ei = load_vocabs()
    # run(eng_vocab, ita_vocab, dict_ei)
    # test()
    example()
