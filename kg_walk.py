##################################
# cidoc2vec file
# author: Hassan El-Hajj
# Max Planck Institute for the History of Science
# Sphaera Project
# Department I
##################################

import pickle
import kg_embedding
import argparse
import pandas as pd
import query_information
import pickle
import os


def main():
    parser = argparse.ArgumentParser(description= 'Walks across CIDOC-CRM KG')
    parser.add_argument('-g', '--graph', type = str, help= 'path to the input graph - RDF TTL format', required= True)
    parser.add_argument('-df', '--data', type = str, help = 'path to JSON or CSV with Data', required= True)
    parser.add_argument('-q', '--query', type = str, help = 'path to SPARQL query that queries all connection from chosen entity', required=True)
    parser.add_argument('-d', '--depth', type = int, help = 'int : sets the depth of walk from central entity', default=15)
    parser.add_argument('-i', '--iteration', type = int,  help = 'int : number of walk interation per entity', default= 200)
    parser.add_argument('-r', '--restart', type = bool, help = 'defaul False, if True, walk restart when it reaches a leaf', default= False)
    parser.add_argument('-o', '--output', type = str, help = 'path to output folder to save walks', required= True)
    
    args =  parser.parse_args()

    # Check format and read data
    print('Loading data...')

    data_format = str(args.data).split('.')
    data_format = data_format[-1]
    if data_format == 'csv':
        df = pd.read_csv(args.data)
    elif data_format == 'json':
        df = pd.read_json(args.data, lines=True)
    else:
        print('Unsupported data format - Please use CSV or JSON files for data entry')
        exit()
    
    # check format and read file 
    print('Loading Graph...')
    data_format = str(args.graph).split('.')
    data_format = data_format[-1]
    if data_format == 'ttl':
        g = query_information.load_rdf(args.graph)
        
    else:
        print('Unsupported graph format - Please use a TTL file for the graph entry')
        exit()

    print('Data and Graph Loaded')

    # check if output path exists
    # to be added

    
    # initiate walks
    print('Starting Walks')
    allitem_walks = []
    i = 0
    for ind, row in df.iterrows():
        print('--------------------------- Processing Item :', i , '----------------------------')
        oneitem_walks = kg_embedding.branch_walk_rdf(g, args.query, row['item'], args.depth, args.iteration, with_restart=args.restart)
        allitem_walks.append(oneitem_walks)
        i = i + 1
    

    print('Saving Results...')
    walks_pickle = args.output + '/walks_data.pickle'
    with open(walks_pickle, 'wb') as f:
        pickle.dump(allitem_walks, f)





if __name__ == "__main__":
    main()
    