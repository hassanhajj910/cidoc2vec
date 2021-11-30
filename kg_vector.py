##################################
# cidoc2vec file
# author: Hassan El-Hajj
# Max Planck Institute for the History of Science
# Sphaera Project
# Department I
##################################

import pickle
import pandas as pd
import numpy as np
import kg_embedding
import argparse
import query_information

def main():
    parser = argparse.ArgumentParser(description= 'Vectorise walks across CIDOC-CRM KG')
    parser.add_argument('-g', '--graph', type = str, help= 'path to the input graph - RDF TTL format', required= True)
    parser.add_argument('-df', '--data', type = str, help = 'path to binary pickle file with stored walks', required= True)
    parser.add_argument('-q', '--query', type = str, help = 'path to query returning ID of each entity in graph', required = True)
    parser.add_argument('-em', '--embedding', type= int, help = 'Embedding vector size, defaults to 32', default = 32)
    parser.add_argument('-win', '--window', type = int, help = 'Doc2Vec window size', default = 5)
    parser.add_argument('-e', '--epoch', type = int, help = 'Training epochs, default 200', default = 200)
    parser.add_argument('-r', '--reduce', choices=['none', 'pca', 'tsne'], help = 'In order to plot the results, you can reduce results', default='none')
    parser.add_argument('-o', '--output', type = str, help = 'output path to save model', required= True)

    args =  parser.parse_args()

    print('Loading Data...')
    walks = kg_embedding.load_walks(args.data)

    print('Loading graph...This may take some time...')
    g = query_information.load_rdf(args.graph)
    

    tagged_data, tags = kg_embedding.make_taggedSentence(args.query, walks, g)

    print('Training Model...This may take some time...')
    model = kg_embedding.train_doc2vec(tagged_data, args.embedding, args.window, min = 1, e = args.epoch)
    
    print('Training Completed !')


    data, plot_label = kg_embedding.process_data(model, tags)

    if args.reduce == 'none':
        sent_vec, label = [], []
        for i in range(len(data)):
            label.append(plot_label[i])
            sent_vec.append(data[i])
        
        df = pd.DataFrame()
        df['id'] = label
        df['embedding'] = sent_vec

        print('Writing data...')
        csv_out = args.output + '/embeddings.csv'
        df.to_csv(csv_out, index=False)

        data_pickle = args.output + '/embedding_data.pickle'
        label_pickle = args.output + '/embedding_labels.pickle'
        with open(data_pickle, 'wb') as f:
            pickle.dump(data, f)
        with open(label_pickle, 'wb') as f:
            pickle.dump(plot_label, f)



    elif args.reduce == 'pca':

        print('Reducing dimenstions...')
        reduced_data = kg_embedding.pca_reduce(data, n = 2)
        sent_vec, label, reduced_list = [], [], []
        for i in range(len(data)):
            label.append(plot_label[i])
            reduced_list.append(reduced_data[i])
            sent_vec.append(data[i])
        
        df = pd.DataFrame()
        df['id'] = label
        df['embedding'] = sent_vec
        df['reduced_embedding'] = reduced_list


        print('Writing data...')
        csv_out = args.output + '/embeddings.csv'
        df.to_csv(csv_out, index=False)

        data_pickle = args.output + '/embedding_data.pickle'
        label_pickle = args.output + '/embedding_labels.pickle'
        reduced_pickle = args.output + '/reduced_embedding.pickle'
        with open(data_pickle, 'wb') as f:
            pickle.dump(data, f)
        with open(label_pickle, 'wb') as f:
            pickle.dump(plot_label, f)
        with open(reduced_pickle, 'wb') as f:
            pickle.dump(reduced_data, f)


    elif args.reduce == 'tsne':

        print('Reducing dimenstions...')
        reduced_data = kg_embedding.tsne(data, n = 2)
        sent_vec, label, reduced_list = [], [], []
        for i in range(len(data)):
            label.append(plot_label[i])
            reduced_list.append(reduced_data[i])
            sent_vec.append(data[i])
        
        df = pd.DataFrame()
        df['id'] = label
        df['embedding'] = sent_vec
        df['reduced_embedding'] = reduced_list


        print('Writing data...')
        csv_out = args.output + '/embeddings.csv'
        df.to_csv(csv_out, index=False)


        data_pickle = args.output + '/embedding_data.pickle'
        label_pickle = args.output + '/embedding_labels.pickle'
        reduced_pickle = args.output + '/reduced_embedding.pickle'
        with open(data_pickle, 'wb') as f:
            pickle.dump(data, f)
        with open(label_pickle, 'wb') as f:
            pickle.dump(plot_label, f)
        with open(reduced_pickle, 'wb') as f:
            pickle.dump(reduced_data, f)

        

    else:
        print('Invalide dimension reduction value, expected: none , pca, tsne')


    model_pickle = args.output + '/model.pickle'
    with open(model_pickle, 'wb') as f:
        pickle.dump(model, f)

    fig_out = args.output + '/entities_plot.jpg'
    kg_embedding.plot_embedded(reduced_data, plot_label, fig_out)

    print('Writing data completed !')
    print('Exiting')



if __name__ == "__main__":
    main()
    



