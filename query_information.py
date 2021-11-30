##################################
# cidoc2vec file
# author: Hassan El-Hajj
# Max Planck Institute for the History of Science
# Sphaera Project
# Department I
##################################

from SPARQLWrapper import SPARQLWrapper, JSON, DIGEST, POST, SPARQLExceptions
import pandas as pd
from pathlib import Path
import numpy as np
import time
from http import client
import rdflib


def stringify(file:str, includes_var = False, var_query = None, var = None):
    """function to just creat a nice string for a variable SPARQL Query.
    var_query is a list of indicating the variables that should be replaced in the string query.
    var represents the variable that should replace var_query. """
    if includes_var == False:
      with open(file,'r') as f:
        sp_q = f.read()

    elif includes_var == True:
      with open(file, 'r') as f:
        sp_q = f.read()

      for j in range(len(var_query)): 
        sp_q = sp_q.replace(str(var_query[j]), str(var[j]))
    return sp_q

def load_rdf(file:str):
    g = rdflib.Graph()
    g.parse(file)
    print('---- File Loaded ----')
    return g


def query_rdf(g, query, queried_var:list):
    res = g.query(query)
    # make empty lists
    res_list = [[] for _ in range(len(queried_var))]
    for row in res:
        for i in range(len(queried_var)):
            res_list[i].append(str(row[queried_var[i]]))
    res_final = []
    for l in res_list:
        res_final.append(l)


    return res_final


