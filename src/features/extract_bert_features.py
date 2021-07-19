import numpy as np
import glob
import os
import timeit
import copy
import pickle
import argparse

from utils.read_file import read_wm_essays
from features.bert_features import bert_embedding_individuals

import torch
from transformers import BertTokenizer,BertModel,BertForPreTraining,BertForQuestionAnswering


def get_sent_labels(token_list):
    sent_labels, sentences, sent_start = [], [], 0
    for i, line in enumerate(token_list):
        if line == '\n':
            sentences.append(sent_labels)
            sent_labels = []
        else:        
            token, label = line.rstrip().split()
            sent_labels.append(label)
    return sentences

def get_sent_tokens(token_list):
    sent_tokens, sentences, sent_start = [], [], 0
    for i, line in enumerate(token_list):
        if line == '\n':
            sentences.append(sent_tokens)
            sent_tokens = []
        else:        
            token, label = line.rstrip().split()
            sent_tokens.append(token)
    return sentences

def sent2features(sent_emb):
    features = []

    for word_emb in sent_emb:
        word_features = {}
        if len(word_emb.shape) > 0:
            for i in range(word_emb.shape[0]):
                word_features['bert_features_{}'.format(i)] = float(word_emb[i])
        else:
            word_features['bert_features_0'] = float(word_emb)
            
        features.append(copy.deepcopy(word_features))
        del word_features
    
    return features



def main():
    parser = argparse.ArgumentParser(description='Feature Extraction for WM data')
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the four column .tsv files with header 'sentence_id    token_id    token   label' in the first line.")
    parser.add_argument("--bert_model",
                        default='bert-base-cased',
                        type=str,
                        help="bert model to be used for embeddings extraction")
    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    bert_model = BertModel.from_pretrained(args.bert_model, output_hidden_states=True)

    _, _, _, _, essay_str_sent = read_wm_essays(args.data_dir)
    sentences = [sent for essay_sent in essay_str_sent for sent in essay_sent]
    print('number of sentences',len(sentences))

    print('embeddings will be exported to: ', os.path.join(args.data_dir, 'features/'))
    embeddings = bert_embedding_individuals(os.path.join(args.data_dir), sentences, tokenizer, bert_model)

    print('\nembeddings extraction done, now creating sequences of feature dicts to be used in the CRF model.')
    features = [sent2features(sent) for sent in embeddings]
    pickle.dump(features, open(os.path.join(os.path.join(args.data_dir, 'features/'), 'embeddings.p'), 'wb'))

    print('Done, exported two files: \nembeddings.p (dictionary-based version, much bigger in size, but needed by the CRF model) \nembeddings.bert.npy (feature vectors only)')


if __name__ == '__main__':
    main()





