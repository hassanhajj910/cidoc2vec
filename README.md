# Cidoc2vec

This is the accompanying code to the CIDOC2VEC algorithm, published in the paper titled, 
" CIDOC2VEC: Extracting Information from Atomized CIDOC-CRM Humanities Knowledge Graphs " _Information_ 12(21), 503, https://doi.org/10.3390/info12120503.

```
Cite as:
@Article{info12120503,
AUTHOR = {El-Hajj, Hassan and Valleriani, Matteo},
TITLE = {CIDOC2VEC: Extracting Information from Atomized CIDOC-CRM Humanities Knowledge Graphs},
JOURNAL = {Information},
VOLUME = {12},
YEAR = {2021},
NUMBER = {12},
ARTICLE-NUMBER = {503},
URL = {https://www.mdpi.com/2078-2489/12/12/503},
ISSN = {2078-2489},
DOI = {10.3390/info12120503}
}
```

## Notes on Running CIDOC2VEC
If you are using CIDOC2VEC Knowledge Graph models, you should be able to infer extra information on the some of the main entities stored in your Knowledge Graph. 

To use the following repository, you should provide URI list, similar to the one stored in ./data, of all the Main Entities. These will be the focus on the analysis and the starting entity of each walk. You should also have a .ttl file of your Knowledge Graph. For ID generation, you should replace ./queries/get_id.sparql with the appropriate SPARQL query that satisfies your dataset.
ID queries will be changed in near future so that it is not SPARQL dependent. 

Run kg_walk.py to generate walks across your Knowledge Graph. More details to follow here. 
Run kg_vector.py to generate vector representations of each of the texts generated from the walks. More details to follow here.

both functions have informative --help tags.

The analysis of the results depends on the domain expert, i.e. you.


## To follow
 - More details Readme and a short summary of the main functions
 - demo notebook
 - dummy dataset 




