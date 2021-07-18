import pickle
from collections import defaultdict

def get_sent_types(sents_labels, typ, n_sent):
    '''Returns counts of input sentence type + article length if less than n_sent (otherwise returns n_sent)'''
    typ_counts = [1 if label == typ else 0 for _, label in sents_labels]
    
    article_length_list = [1 for _, _ in sents_labels]

    if len(typ_counts) == n_sent:
        return typ_counts, article_length_list
    
    elif len(typ_counts) > n_sent:
        return typ_counts[:n_sent], article_length_list[:n_sent]
    
    else:
        for i in range(n_sent - len(typ_counts)):
            typ_counts.append(0)
            article_length_list.append(0)
        return typ_counts, article_length_list
        

def sent_types_counts(sentences, predictions, n_sent, add_one_pred=True):
    'Retruns article splitted along with their sentences labels'
    
    if add_one_pred:
        predictions.insert(0,'0\n')
    assert len(sentences) == len(predictions)
    
    article, article_splits, article_lengths = [], [], []
    claim_norm_counts, premise_norm_counts, none_norm_counts = [], [], []
    for i, (sent, pred) in enumerate(zip(sentences, predictions)):
        if sent == 'ARTICLE_SPLIT_LINE\t0\n':
            article_splits.append(article)
            none_count, none_article_length = get_sent_types(article, 0, n_sent)
            claim_count, claim_article_length = get_sent_types(article, 1, n_sent)
            premise_count, premise_article_length = get_sent_types(article, 2, n_sent)
            assert sum(none_article_length) == sum(claim_article_length) == sum(premise_article_length)

            none_norm_counts.append(none_count)
            claim_norm_counts.append(claim_count)
            premise_norm_counts.append(premise_count)
            article_lengths.append(none_article_length)

            article = []
        else:
            article.append(( sent.split('\t')[0], int(pred.rstrip()) ))
    
    return claim_norm_counts, premise_norm_counts, none_norm_counts, article_lengths


# def get_counts_per_article_type_archive(sentences, predictions, article_labels, publisher, n_sent=60):
    
#     claim_norm_counts, premise_norm_counts, none_norm_counts = sent_types_counts(sentences, predictions, n_sent)
    
#     news_counts, op_counts = defaultdict(dict), defaultdict(dict)
#     for i, label in enumerate(article_labels):
#         if label == 1:
#             if publisher[i] not in op_counts.keys():
#                 op_counts[publisher[i]] = defaultdict(list) 
#             op_counts[publisher[i]]['claim'].append(claim_norm_counts[i])
#             op_counts[publisher[i]]['premise'].append(premise_norm_counts[i])
#             op_counts[publisher[i]]['none'].append(none_norm_counts[i])
#         else:
#             print(publisher[i])
#             if publisher[i] not in news_counts.keys():
#                 news_counts[publisher[i]] = defaultdict(list) 
#             news_counts[publisher[i]]['claim'].append(claim_norm_counts[i])
#             news_counts[publisher[i]]['premise'].append(premise_norm_counts[i])
#             news_counts[publisher[i]]['none'].append(none_norm_counts[i])

#     return news_counts, op_counts


def get_counts_per_article_type(sentences, predictions, article_labels, publisher, n_sent=60):
    
    claim_norm_counts, premise_norm_counts, none_norm_counts, article_lengths = sent_types_counts(sentences, predictions, n_sent)
    
    news_counts, op_counts = {}, {}
    news_counts['claim'], news_counts['premise'], news_counts['none'], news_counts['length'] = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    op_counts['claim'], op_counts['premise'], op_counts['none'], op_counts['length'] = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)

    for label, pub, claim, premise, none, length in zip(article_labels, publisher, claim_norm_counts, premise_norm_counts, none_norm_counts, article_lengths):
        if label == 1:
            op_counts['claim'][pub].append(claim)
            op_counts['premise'][pub].append(premise)
            op_counts['none'][pub].append(none)
            op_counts['length'][pub].append(length)
        else:
            pass
            news_counts['claim'][pub].append(claim)
            news_counts['premise'][pub].append(premise)
            news_counts['none'][pub].append(none)
            news_counts['length'][pub].append(length)

    return news_counts, op_counts

def main(dataset='train', n_sent=60):

    train = pickle.load(open('/Users/tariq/Downloads/Bloomberg_Editorial_Classifier/data/train_test_v2/X_train.pkl','rb'))
    train_labels = pickle.load(open('/Users/tariq/Downloads/Bloomberg_Editorial_Classifier/data/train_test_v2/y_train.pkl','rb'))
    train_labels = train_labels.label
    train_publishers = train.source_name
    train_sent = open('/Users/tariq/Downloads/Bloomberg_Editorial_Classifier/data_tsv/train_sent/dev.tsv').readlines()
    train_pred = open('/Users/tariq/Downloads/Bloomberg_Editorial_Classifier/data_tsv/train_sent/predictions_editorial-claim-premise-bert.txt').readlines()
    train_news_counts, train_op_counts = get_counts_per_article_type(train_sent, train_pred, train_labels, train_publishers, n_sent)
    
    dev = pickle.load(open('/Users/tariq/Downloads/Bloomberg_Editorial_Classifier/data/train_test_v2/X_test.pkl','rb'))
    dev_labels = pickle.load(open('/Users/tariq/Downloads/Bloomberg_Editorial_Classifier/data/train_test_v2/y_test.pkl','rb'))
    dev_labels = dev_labels.label
    dev_publishers = dev.source_name
    dev_sent = open('/Users/tariq/Downloads/Bloomberg_Editorial_Classifier/data_tsv/dev_sent/dev.tsv').readlines()
    dev_pred = open('/Users/tariq/Downloads/Bloomberg_Editorial_Classifier/data_tsv/dev_sent/predictions_editorial-claim-premise-bert.txt').readlines()
    dev_news_counts, dev_op_counts = get_counts_per_article_type(dev_sent, dev_pred, dev_labels, dev_publishers, n_sent)

    # test = pickle.load(open('/Users/tariq/Downloads/Bloomberg_Editorial_Classifier/data/collected/metro winnipeg - extra test/dev.pkl','rb'))
    # test = pickle.load(open('/Users/tariq/Downloads/Bloomberg_Editorial_Classifier/data/collected/metro winnipeg - extra test/dev.pkl','rb'))
    # test_labels = test.label
    # test_sent = open('/Users/tariq/Downloads/Bloomberg_Editorial_Classifier/data_tsv/test_sent/dev.tsv').readlines()
    # test_pred = open('/Users/tariq/Downloads/Bloomberg_Editorial_Classifier/data_tsv/test_sent/predictions_editorial-claim-premise-bert.txt').readlines()
    # test_news_counts, test_op_counts = get_counts_per_article_type(test_sent, test_pred, test_labels)
    
    if dataset == 'train':
        return {'news': train_news_counts, 'op':train_op_counts}
    elif dataset == 'dev':
        return {'news': dev_news_counts, 'op':dev_op_counts}
    else :
        print('dataset has be either train or dev')
        return



if __name__ == "__main__":
    main()