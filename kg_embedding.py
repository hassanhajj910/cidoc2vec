##################################
# cidoc2vec file
# author: Hassan El-Hajj
# Max Planck Institute for the History of Science
# Sphaera Project
# Department I
##################################



# Query walk should walk along the KG and query consecutive patterns.
# query walk should be up to a predetermined length, and should take into 
# consideration repeating info. 
import pickle
from pandas.core.algorithms import unique
import query_information
import numpy as np
import pandas as pd
import rdflib
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import argparse


def get_theta(g, node, lastbeta):
    """Calculate theta
    Input:
    g : Graph from which to query
    node: Current node to evaluate on.
    lastbeta: list of beta, which include the last weight to be used in theta calculate. 
    Output:
    Theta: Importance score."""
    v = [node]
    q = query_information.stringify('./queries/get_indegree_theta.sparql', includes_var=True, var_query=['{ITEM}'], var = v)
    res = query_information.query_rdf(g, q, queried_var=['h', 'r', 't'])
    eta = 1.e-7
    theta = np.log(len(res[1]) + eta)
    theta = lastbeta * theta
    return theta

def get_beta_reverse(g, node:str):
    """Calculate Beta considering reversed connectins
    Input:
    g: Graph to query
    node: Current node
    Ouput:
    rel_dic: dictionary containing each relations and its associated weight
    ww: weight list
    rel1: relation list"""
    
    # Select node
    item = [node]

    # query incoming and outgoing edges
    q = query_information.stringify('./queries/get_outdegree.sparql', includes_var=True, var_query=['{ITEM}'], var = item)
    res = query_information.query_rdf(g, q, queried_var=['h', 'r', 't', 'r2', 't2'])
    # local out degree sum
    local_out1 = len(res[0])
    q2 = query_information.stringify('./queries/get_indegree.sparql', includes_var=True, var_query=['{ITEM}'], var = item)
    res2 = query_information.query_rdf(g, q2, queried_var=['t', 'r' , 'h', 'r2' , 'h2' ])
    # local out from incoming edges. 
    local_out2 = len(res2[0])
    
    # Local sum
    local_out = local_out1 + local_out2
    # get all relations as reverse relations are considered.
    rel1 = list(set(list(res[1])))
    relprime = list(set(list(res2[1])))
    rel1.extend(relprime)
    # unique relations
    rel1 = list(set(rel1))

    # transform to arr
    resar = np.array(res).T
    resar2 = np.array(res2).T
    resar = np.concatenate((resar, resar2), axis = 0)

    rel_dict = dict()
    rel_dict = dict()

    ww, ww2 = [], []
    for r1 in rel1:
        # calculate weight
        rel2 = resar[resar[:,1] == r1]
        # rel2 count
        dout = rel2.shape[0]
        weight = (dout)/local_out
        rel_dict[r1] = weight
        ww.append(weight)


    return rel_dict, ww, rel1

def get_beta(g, node:str):
    """Calculate Beta with only outgoing connections.
    Input:
    g: Graph to query
    node: Current node
    Ouput:
    rel_dic: dictionary containing each relations and its associated weight
    ww: weight list"""

    # similar to reverse beta comments. 

    item = [node]
    q = query_information.stringify('./queries/get_outdegree.sparql', includes_var=True, var_query=['{ITEM}'], var = item)
    res = query_information.query_rdf(g, q, queried_var=['h', 'r', 't', 'r2', 't2'])
    local_out = len(res[0])
    # first order relations.
    rel1 = list(set(list(res[1])))
    # transform to arr
    resar = np.array(res).T
    rel_dict = dict()
    ww = []
    for r1 in rel1:
        # calculate weight
        rel2 = resar[resar[:,1] == r1]
        # rel2 count
        dout = rel2.shape[0]
        weight = (dout)/local_out
        rel_dict[r1] = weight
        ww.append(weight)

    return rel_dict, ww

def RSW(res, g, head:list, relation:list, tail:list, betalist:list, walktype:str, threshold = 0):
    """Directs RSW KG walk
    --- Input ---
    Res: result of walk stance
    g: graph
    head,relation,tail : containers of the complete walk entities
    betalist: list of last connection weights
    walkstype: random or weighed
    threshold: Theta threshold. 
    --- Return ---
    head,relation,tail : containers with the new direction of the walk.
    reverse: flag about reverse. if true it is reverse walk.
    betalist: filled betalist. 
    """

    if walktype == 'random':
        # Initialte reverse Flag to false. 
        reverse = False
        # gets random number bsetween 0 and max relations. 
        rand_relation = np.random.randint(0, len(res[0]))
        # append corresponding data to given lists. 
        head.append(res[0][rand_relation])
        relation.append(res[1][rand_relation])
        tail.append(res[2][rand_relation]) 
        # Random choice, add 1 weight to Beta. 
        betalist.append(1)
        # returns augmented lists. 
        return head, relation, tail, reverse, betalist


    elif walktype == 'weighed':

        # Check theta. 
        theta = get_theta(g, res[0][0], betalist[-1])
        ############################
        # if below threshold. just continue with weighed walk.
        if theta < threshold: 
            reverse = False

            weight_dic, ww = get_beta(g, res[0][0])
            # 
            if len(ww) > 0:
                weight_list = []
                for r in list(set(res[1])):
                    weight_list.append(weight_dic[r])
                
                toselect = np.arange(0, len(weight_list), 1)
                draw = np.random.choice(toselect, p = weight_list)

                betalist.append(weight_list[draw])

            else:
                draw = np.random.randint(0, len(res[0]))
                betalist.append(1)

            head.append(res[0][draw])
            relation.append(res[1][draw])
            tail.append(res[2][draw])
            # returns augmented lists. 
            return head, relation, tail, reverse, betalist
        
        # If above threshold - consider incoming edges, and perform backwards walk to the main Entity. 
        if theta >= threshold:
            reverse = False
            weight_dic, ww, rel_list = get_beta_reverse(g, res[0][0])

            
            if len(ww) > 0:
                weight_list = []
                for r in list(set(rel_list)):
                    weight_list.append(weight_dic[r])
                
                toselect = np.arange(0, len(weight_list), 1)

                draw = np.random.choice(toselect, p = weight_list)
                betalist.append(weight_list[draw])


                headrev, relationrev, tailrev = [], [], []
                if rel_list[draw] == relation[-1]:
                    # print(' -----------------  Same relation ----------------')
                    reverse = True
                    # means it is the same as relation coming in. 
                    # perform walks tracing back the initial walks coming. 

                    for q in range(0, len(head)):
                        if q == 0:
                            obj_tail = tail[-1]
                            obj_rel = relation[-1]
                        else:
                            obj_tail = res_rev[2][0]    
                            obj_rel  = relation[-q-1]

                        if obj_tail[:4] != 'http':
                            break
                        q_rev = query_information.stringify('./queries/get_reverse.sparql', includes_var= True, var_query=['{object_tail}', '{object_relation}'], var = [obj_tail, obj_rel])
                        res_rev = query_information.query_rdf(g, q_rev, queried_var= ['h', 'r', 't'])
                        

                        if len(res_rev[0]) == 0:
                            break

                        headrev.append(res_rev[2][0])
                        relationrev.append(res_rev[1][0])
                        tailrev.append(res_rev[0][0])
                else:
                    weight_dic, ww = get_beta(g, res[0][0])
                    if len(ww) > 0:
                        weight_list = []
                        for r in list(set(res[1])):
                            weight_list.append(weight_dic[r])
                        
                        toselect = np.arange(0, len(weight_list), 1)
                        draw = np.random.choice(toselect, p = weight_list)
                        betalist.append(weight_list[draw])

                    head.append(res[0][draw])
                    relation.append(res[1][draw])
                    tail.append(res[2][draw])

                head.extend(headrev)
                relation.extend(relationrev)
                tail.extend(tailrev)
                        

            return head, relation, tail, reverse, betalist
   
def branch_walk_rdf(rdf_graph, init_sparql:str, base:str, depth:int, iteration = 1, with_restart=False):
    """Initiate and walk through the KG starting from Base - the walk is initiated on a RDF - TTL Graph.
    --- Input ---
    rdf_graph: loaded RDF TTL Graph storing the KG database. 
    init_query: file with the SPARQL query that queries all connection to current entity.
    base: base URI, around which the database is centered
    depth: depth of walk from base to be collected
    interation : number of iterations per walk, as int.
    with_restart: default False. When True, when a walk reaches a leaf, the same sentence restarts the walk from base entitiy.
    --- Return ---  
    Walks: Array 2D of walks across KG"""
    # create a storage space for h,r,t
    walks = np.zeros((iteration, 3), dtype=object)
    betas = []

    # go through all iterations. 
    for i in range(iteration):
        if i > 0 and i % 100 == 0:
            print('---- --- WALK ITERATION: ', i,)            # print every 100 iterations. 
        
        #branch walk storage lists.
        h, r, t = [], [], []

        # query initial round.
        # query starts from base URI fed to function.
        init_q = query_information.stringify(init_sparql, includes_var=True, var_query=['{object}'], var = [base])
        init_res = query_information.query_rdf(rdf_graph, init_q, queried_var= ['h', 'r', 't'])

        # choose random relation to tail. 
        h, r, t, flag, betas = RSW(init_res, rdf_graph, h, r, t, betas, walktype = 'random')

        # branch walk
        restart = False                 # default False restart.

        # walk limited to chosen depth as max value. Could be cut short. 
        for j in range(depth):

            if restart == True:         # in case of restart base defaults to init base.
                base_var = h[0] 
            else:
                base_var = t[-1]        # in case on none-restart, base is last tail.

            if base_var[:4] != 'http':  # Check that base are URI. Otherwise this is a leaf. Leafs are often STR or INT (at least in SPHAERA data)
                if with_restart == True:
                    restart = True
                    continue
                else:
                    break

            # query and walk along branch at each iteration move one step. 
            query = query_information.stringify(init_sparql, includes_var=True, var_query=['{object}'], var = [base_var])
            res = query_information.query_rdf(rdf_graph, query, queried_var= ['h', 'r', 't'])

            # if query resturns nan. result does not contain data. restart walk.
            if len(res[0]) == 0:
                if with_restart == True:
                    restart = True
                    continue
                else:
                    break
            
            # After imitial step, start weight RSW walks with chosen threshold. 
            # h, r, t, flag, betas = RSW(res, rdf_graph,  h, r, t, betas, walktype='weighed', threshold= 2.5)
            h, r, t, flag, betas = RSW(res, rdf_graph,  h, r, t, betas, walktype='random')
            restart = False             # defaults restart back to False. 
            # if flag is True means that walk was restarted and thus walk should NOT continue. 
            if flag == True:
                break
        # store as object array. 
        walks[i][0] = h
        walks[i][1] = r
        walks[i][2] = t

    return walks

def branch_walk(init_sparql:str, base:str, depth:int, iteration = 1, with_restart=False):
    """Initiate and walk through the KG starting from Base.
    --- Input ---
    init_query: file with the SPARQL query that queries all connection to current entity.
    base: base URI, around which the database is centered
    depth: depth of walk from base to be collected
    interation : number of iterations per walk, as int.
    with_restart: default False. When True, when a walk reaches a leaf, the same sentence restarts the walk from base entitiy.
    --- Return ---  
    Walks: Array 2D of walks across KG"""

    # create a storage space for h,r,t
    walks = np.zeros((iteration, 3), dtype=object)

    # loop over each iteration
    for i in range(iteration):
        print(' ------ WALK ITERATION: ', i, '-------')

        #branch walk storage lists.
        h, r, t = [], [], []

        # query initial round.
        # query starts from base URI fed to function.
        init_q = query_information.stringify(init_sparql, includes_var=True, var_query=['{object}'], var = [base])
        init_res = query_information.query_local(init_q, queried_var= ['h', 'r', 't'], query_type='select')

        # choose random relation to tail. 
        h, r, t = RSW(init_res, h, r, t)

        # branch walk
        restart = False                 # default False restart.
        for j in range(depth):

            if restart == True:         # in case of restart base defaults to init base.
                base_var = h[0] 
            else:
                base_var = t[-1]        # in case on none-restart, base is last tail.

            if base_var[:4] != 'http':  # Check that base are URI. Otherwise this is a leaf. 
                if with_restart == True:
                    restart = True
                    continue
                else:
                    break
            
            query = query_information.stringify(init_sparql, includes_var=True, var_query=['{object}'], var = [base_var])
            res = query_information.query_local(query, queried_var= ['h', 'r', 't'], query_type='select')

            # if query resturns nan. This is usually Erlangen or other URI, and do not contain data. restart walk.
            if res[0][0] is np.nan:
                if with_restart == True:
                    restart = True
                    continue
                else:
                    break
            
            # choose random walk from the resulting relations, and add to storage lists. 
            h, r, t = RSW(res, h, r, t)
            restart = False             # defaults restart back to False. 
            
        
        # store as object array. 

        walks[i][0] = h
        walks[i][1] = r
        walks[i][2] = t
        # print(walks[:][:])
        print('---------  END WALK  -------')
    return walks

def load_walks(file:str):
    """Loads a pickle file with walks - used when retreiving saved data.
    --- Input ---
    file: pickle file with walks from branch_walk_rdf
    --- Return ---
    walks: array from branch_walk_rdf"""
    with open(file, 'rb') as f:
        walks = pickle.load(f)
    return walks

def make_taggedSentence(tag_sparql:str, walks, rdf_graph):
    """Prepares sentences for doc2vec
    --- Input ---
    tag_sparql : sparql query retrieving info (such as ID) of each entity.
    walk : resulting walks from branch_walk_rdf
    rdf_graph : loaded RDF graph
    --- Return ---
    tagged_data : list of sentences with their appriate tags"""

    # containing lists
    sentence = []
    sent_tag = []
    # count
    q = 0
    # each walk contains many sub_walks
    for w in walks:
        # select main object - the main object of the walk is usually the first item in the walk, hence the triple 0 indexing.
        obj = w[0][0][0]
        # get the information relating to that object from the graph. Here ID. 
        query = query_information.stringify(tag_sparql, includes_var=True, var_query=['{object}'], var = [obj])
        res = query_information.query_rdf(rdf_graph, query, queried_var=['id'])
        id = res[0][0]
        # current tag is ID. 
        tag = id
        # sub_walks contain the different sentences from the same document/entity. 
        for sub in w:
            # w represent the whole doc assiciated with a book.
            # sub is a single sentence from book doc.
            h = sub[0]
            r = sub[1]
            t = sub[2]
            # for single sentence
            sng_sent = []
            for i in range(len(h)):
            # to construct sentence we follow the logic h/r/t/r/t and so on.
                if i == 0:
                    sng_sent.append(h[i])
                sng_sent.append(r[i])
                sng_sent.append(t[i])
            sentence.append(sng_sent)
            sent_tag.append(tag)
        q = q + 1   # iterator

    # make tagged data (GENSIM required)
    tagged_data = []
    for i in range(len(sentence)):
        tagged_data.append(TaggedDocument(sentence[i], [sent_tag[i]]))

    return tagged_data, sent_tag

def train_doc2vec(data, emb_size, win, min, e, ):
    """Train doc2vec
    --- Input ---
    data : walk data array
    emb_size : embedding vector size
    win : Window size
    min : threshold of minimum occurances
    e : epochs
    --- Return ---
    model : doc2vec trained model"""
    model = Doc2Vec(data, vector_size = emb_size, window = win, min_count = min, epoch = e)
    return model

def process_data(model, sentence_tag:list):
    """Process data by creating unique tags, as well as assigning the model data into an array with approriate labels 
    --- Input ---
    model: trained doc2vec model
    sentence_tag: list of sentence tag representing every sentence in the model.
    --- Return ---
    data : resulting embedding vectors for each entity
    plot_label : list of labels (from sentence tags) that makes it easier to plot later"""
    # create container
    uniq_tag = list(set(sentence_tag))
    temp = model.docvecs[uniq_tag[0]]
    # get storage dim based on embedding results of model.
    temp = len(temp)
    data = np.zeros((len(uniq_tag), temp))
    plot_label = []
    sent_vec = []
    for i in range(len(uniq_tag)):
        data[i][:] = model.docvecs[uniq_tag[i]]
        sent_vec.append(data[i])
        plot_label.append(uniq_tag[i])
    
    data_df = pd.DataFrame()
    data_df['id'] = plot_label
    data_df['embedding'] = sent_vec

    return data, plot_label

def kmean_colorplot(data, n:int):
    """
    Helper function to plot Kmeans clusters
    --- Input --- 
    Data: array generated from process_data
    n: kmean n_comp
    --- Return ---
    km_plotcolor_ a list of that helps plot different clusters in different colours."""

    # supports only up to 5 colors now. 
    if n > 5:
        print('up to n = 5 is supported...n is set to 5')
        n = 5

    colorlist = ['b', 'r', 'g', 'm', 'y', 'k']
    kmean = KMeans(n_clusters = n, random_state= 42).fit(data)
    km_lab = kmean.labels_

    km_plotColor = []
    for i in range(len(km_lab)):
        km_plotColor.append(colorlist[km_lab[i]])
    return km_plotColor

def pca_reduce(data, n = 2):
    """Reduces embedding data from higher dimention to n dim. 
    --- Input ---
    Data: array generated from process_data with sentence embeddings.
    n: pca n_comp
    --- Returns ---
    pca_data :  reduced data"""
    # standardise data
    std_data = StandardScaler().fit_transform(data)
    pca = PCA(n_components=n)
    pca_data = pca.fit_transform(std_data)
    return pca_data

def tsne(data, n = 2):
    """Prepares data to n dim for visualisation"""
    tsn = TSNE(n_components=2, random_state=42).fit_transform(data)
    return tsn

def plot_embedded(data, uniq_label, output):
    """Plots embedded entity vectors and saves figure in output"""
    # set figure size

    plt.figure(figsize=(30,25))
    plt.scatter(data[:,0], data[:,1])


    for i, txt in enumerate(uniq_label):
            plt.annotate(txt, (data[i,0], data[i,1]))
    
    plt.title('2D Embedding Space')
    plt.xlabel('X')
    plt.ylabel('Y')

    plt.savefig(output, dpi = 300)
