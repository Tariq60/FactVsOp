import torch
from transformers import BertTokenizer,BertModel,BertForPreTraining,BertForQuestionAnswering
import numpy as np
import glob
import os
import pickle

# tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
# bert_model = BertModel.from_pretrained('bert-base-cased',output_hidden_states=True)

def get_individual_token_ids(tokenizer, sentence, T=120):
    
    tokens = tokenizer.tokenize(sentence)
    tokens = ['[CLS]'] + tokens + ['[SEP]']
 
    padded_tokens = tokens +['[PAD]' for _ in range(T-len(tokens))]
    attn_mask = [ 1 if token != '[PAD]' else 0 for token in padded_tokens  ]

    seg_ids = [1 for _ in range(len(padded_tokens))]
    sent_ids = tokenizer.convert_tokens_to_ids(padded_tokens)
#     print("senetence idexes \n {} ".format(sent_ids))

    token_ids = torch.tensor(sent_ids).unsqueeze(0) 
    attn_mask = torch.tensor(attn_mask).unsqueeze(0) 
    seg_ids   = torch.tensor(seg_ids).unsqueeze(0)
        
    
    return tokens, token_ids, attn_mask, seg_ids


def get_embedding(last_1_layer, last_2_layer, last_3_layer, last_4_layer, T=120):

    token_list = []
    
    for index in range(T):
        token = torch.add(last_1_layer[index],last_2_layer[index])
        token = torch.add(token,last_3_layer[index])
        token = torch.add(token,last_4_layer[index])
        #print(token)
        token_mean = torch.div(token, 4.0)
        #print(token_mean)
        token_list.append(token_mean)
        #token_mean.shape

#     print ('Shape is: %d x %d' % (len(token_list), len(token_list[0])))

#     sentence_embedding = torch.mean(torch.stack(token_list), dim=0)
#     print(sentence_embedding.shape)

    return token_list


def get_embedding_from_bert(bert_model, token_ids, attn_mask, seg_ids, num_layers=4, T=120, device='cpu'):
    
    bert_model.eval()

    with torch.no_grad():
        model_outputs = bert_model(token_ids.to(device), attention_mask = attn_mask.to(device), token_type_ids = seg_ids.to(device))
        # model_outputs = bert_model(token_ids, attention_mask = attn_mask, token_type_ids = seg_ids)

    last_4_hidden_states = model_outputs[-1][-num_layers:]
#     print('**********', len(model_outputs), len(model_outputs[-1]), len(last_4_hidden_states))
#     print(token_ids)
    
    last_1_layer = torch.squeeze(last_4_hidden_states[0],dim=0)
    last_2_layer = torch.squeeze(last_4_hidden_states[1],dim=0)
    last_3_layer = torch.squeeze(last_4_hidden_states[2],dim=0)
    last_4_layer = torch.squeeze(last_4_hidden_states[3],dim=0)

    token_list_embedding = get_embedding(last_1_layer, last_2_layer, last_3_layer, last_4_layer, T)
    
    return token_list_embedding[:np.count_nonzero(attn_mask)]




def bert_embedding_individuals(output_path, output_file, sentences, tokenizer, bert_model, T=120, device='cpu'):

#     if not os.path.exists(os.path.join(output_path, 'features/')):
#         os.makedirs(os.path.join(output_path, 'features/'))

    sentence_embeddings = []
    
    for sent_id, sentence in enumerate(sentences):
        # print(sentence)
        if len(sentence)==0:
            print(sent_id, 'empty sentence')
            sentence_embeddings.append([])
            continue

        try:
            if sent_id > 0 and sent_id % 50 == 0:
                print('processed {} sentences'.format(sent_id))
            sent_tokens = sentence.split()
            tkns, token_ids, attn_mask, seg_ids = get_individual_token_ids(tokenizer, sentence, T)
            token_list_embedding = get_embedding_from_bert(bert_model, token_ids, attn_mask, seg_ids, T=T, device=device)

            assert tkns[0] == '[CLS]'
            # print(token_list_embedding[0][0].shape, torch.squeeze(token_list_embedding[0]).shape)
            sentence_embeddings.append(torch.squeeze(token_list_embedding[0]))

        except Exception as e:
            np.save(os.path.join(output_path, '{}_{}.npy'.format(output_file, sent_id)), sentence_embeddings)
            print(e)
            exit()

    
    pickle.dump(sentence_embeddings, open(os.path.join(output_path, '{}.p'.format(output_file)), 'wb'))
    # np.save(os.path.join(output_path, 'features/embeddings.bert.npy'), sentence_embeddings)
    
    return sentence_embeddings
 
 