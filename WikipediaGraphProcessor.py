#!/usr/bin/env python3
# coding: utf-8

import os

import networkx as nx

import community

import igraph as ig
import leidenalg as la

import matplotlib.pyplot as plt
import statistics

from neo4j import GraphDatabase, Driver

import pyspark.ml
from pyspark.sql import SparkSession
from pyspark.ml.feature import IDF, Tokenizer, StopWordsRemover, CountVectorizer
from pyspark.ml.clustering import LDA
from pyspark.sql.functions import udf, col, lower, regexp_replace
from pyspark.sql import types as T

import spacy

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer


# Importing graph
def import_graph(path):
    print('importing graph from :', path)
    directed_g = nx.read_gexf(path, node_type=None, relabel=True)
    undirected_g = directed_g.to_undirected()
    return undirected_g, directed_g


# Verification
def verify_graph(g):
    print('number of nodes : ' + str(len(g)))
    fig, ax = plt.subplots(figsize=(70, 50)) # set size
    nx.draw(g, with_labels=True)
    plt.show()


# Graph cleaning
def clean_graph(graph):
    print('cleaning graph')
    #Degree computation
    nodes_with_degrees = graph.degree
    #mean
    mean_deg = statistics.mean(l[1] for l in nodes_with_degrees)
    #std
    std_deg = statistics.stdev(l[1] for l in nodes_with_degrees)
    #threshold
    threshold = mean_deg + std_deg*std_deg/2 

    #Filtering extremely highly connected nodes
    nodes_to_remove = list(filter(lambda d : d[1] > threshold, nodes_with_degrees))
    n,d = zip(*nodes_to_remove)
    graph.remove_nodes_from(n)
    
    #Filtering isolated nodes
    nodes_with_degrees = graph.degree
    nodes_to_remove = list(filter(lambda d : d[1] == 0, nodes_with_degrees))
    n,d = zip(*nodes_to_remove)
    graph.remove_nodes_from(n)

    print('remaining nodes : ' + str(len(graph)))
    
    return graph


# Keeping only largest connected component
def largest_connected_component(graph):
    print('largest connected component')
    gcc = max(nx.connected_component_subgraphs(graph), key=len)
    print('remaining nodes : ' + str(len(gcc)))
    return gcc


# Partitioning using Louvain
def communities_louvain(graph):
    louvain_communities = community.best_partition(graph, resolution=1)
    louvain_communities_dict = {}
    for key, value in sorted(louvain_communities.items()):
        louvain_communities_dict.setdefault(value, []).append(key)

    print('detcted',len(louvain_communities_dict),'communities')
    
    return louvain_communities_dict

# Partitioning using Leiden
# https://pypi.org/project/leidenalg/
# https://www.nature.com/articles/s41598-019-41695-z
# leiden helpers/adapters
def get_id_to_title(graph):
    tmp = nx.get_node_attributes(graph, 'id')
    d = {}
    for x in tmp:
        d.setdefault(tmp[x], x)
    return d

def nx_to_ig(graph):
    nx.write_graphml(graph,'graph.graphml')
    graphi = ig.read('graph.graphml',format="graphml")
    os.remove('graph.graphml')
    return graphi

def translate_leiden_to_dict(partition, graphi, dictionary):
    nodes = graphi.vs
    res = {}
    for i in range(len(partition)):
        res.setdefault(i, [])
        for v in partition[i]:
            res[i].append(dictionary[nodes[v]['id']])
    return res

def communities_leiden(graph):
    graphi = nx_to_ig(graph)
    partition = la.find_partition(graphi,
                                  la.ModularityVertexPartition) 
    dictionary = get_id_to_title(graph)
    res = translate_leiden_to_dict(partition, graphi, dictionary)
    
    print('detcted',len(res),'communities')

    return res

# categoriy of each partition
# helpers
#get list of categories of a page
def get_categories(page_name):
    c = list()
    with driver.session() as session:
        with session.begin_transaction() as tx:
            for record in tx.run("MATCH (p:Page)-[:BELONGS_TO]->(c:Category) "
                                 "WHERE p.title = {page_name} "
                                 "AND NOT exists((c)-[:BELONGS_TO]->(:Category {title: \'{hc}\'})) "
                                 "RETURN c.title", 
                                 page_name = page_name,
                                 hc = language_mapper[language]):
                c.append(record["c.title"])
    return c

#map each element to frequency in a list    
def count_frequency(my_list): 
    freq = {} 
    for items in my_list: 
        freq[items] = my_list.count(items)
    return freq

#iterate over pages dict partition
def part_category_fetch(key, dic):
    cat = []
    for title in dic[key]:
        cat += get_categories(title)
    return cat

def fetcher(bpd):
    part_cat = {}
    
    for part in sorted(bpd):
        cat = part_category_fetch(part, bpd)
        part_cat.setdefault(part, cat)
    return part_cat


# Fetch categories for each cluster
def fetch_categories(bpd):
    part_cat_dict = fetcher(bpd)    
    return part_cat_dict

def count_all_frequencies(d):
    part_cat_dict_freq = {}
    for e in d:
        cat_map_freq = count_frequency(d[e])
        part_cat_dict_freq.setdefault(e, cat_map_freq)
    return part_cat_dict_freq
 
def find_max_freq(p):
    max_part_cat = {}
    for e in p:
        ls = list(p[e].keys())
        cat = ls[0]
        for x in ls:
            if p[e][cat] < p[e][x]:
                cat = x
        max_part_cat.setdefault(e, cat)
    return max_part_cat

def get_n_largest_communities(c, n):
    x = min(n, len(c))
    print('getting', x, 'largest communities')
    
    l = []
    for e in sorted(c):
        l.append(c[e])
    
    tmp = sorted(l, key=len, reverse=True)[:x]
    
    res = {}
    for i in range(x):
        res.setdefault(i, tmp[i])
    
    length = 0
    for e in res:
        length += len(res[e])
    
    return res

#Load Function
def ld(path, n):
    undirected_g, directed_g = import_graph(path)
    #verify_graph(g)
    gg = largest_connected_component(clean_graph(undirected_g))
    #verify_graph(gg)
    
    #communities_dict = communities_louvain(gg)
    communities_dict = communities_leiden(gg)
    
    communities = get_n_largest_communities(communities_dict, n)
    
    undirected_graph = gg.subgraph([x for y in communities.values() for x in y])
    directed_graph = directed_g.subgraph([x for y in communities.values() for x in y])
        
    print('new number of nodes is :', len(directed_graph))
    
    part_cat_dict = fetch_categories(communities)
    
    return undirected_graph, directed_graph, communities, part_cat_dict

def tokenize(df):
    print('tokenizing')
    tokenizer = Tokenizer(inputCol="categories", outputCol="raw")
    res = tokenizer.transform(df)
    return res
    
def stop_words_remove(df):
    print('stopWords removal')
    remover = StopWordsRemover(inputCol="raw", outputCol="words")
    res = remover.transform(df)
    return res

def lemmatize(df):
    print('lemmatization')
    lemmatizer = WordNetLemmatizer()
    lemmatizer_udf = udf(lambda tokens: [lemmatizer.lemmatize(token) for token in tokens], T.ArrayType(T.StringType()))
    res = df.withColumn("words", lemmatizer_udf("words"))
    return res

def cv_fit(df):
    print('countVectorizer')
    countVectorizer = CountVectorizer(inputCol="words", outputCol="rawFeatures")
    cvmodel = countVectorizer.fit(df)
    return cvmodel

def cv_transform(cvmodel, df):
    res = cvmodel.transform(df)
    return res

def idf(df):
    print('IDF')
    idf_ = IDF(inputCol="rawFeatures", outputCol="features")
    idfModel = idf_.fit(df)
    res = idfModel.transform(df)
    return res
        
def lda_fit(df):
    # Trains a LDA model.
    print('training LDA')
    lda_ = LDA(k=df.count(), maxIter=100)
    ldaModel = lda_.fit(df)
    return ldaModel

def lda_transform(ldaModel, df):
    print('LDA transformation')
    transformed = ldaModel.transform(df)
    #transformed.show()
    return transformed

def show_topic_description(ldaModel, cvmodel):
    topicIndices = ldaModel.describeTopics(maxTermsPerTopic = 5)
    vocabList = cvmodel.vocabulary
    tops = []
    for i,t,w in topicIndices.collect():
        print('Topic %d:' % i)
        entry = []
        for j in range(len(t)):
            entry.append(vocabList[t[j]])
        print(entry)
        tops.append(entry)
        
    return tops
    
def get_topics(communities, tops, transformed):
    cluster_topicDist = sorted(transformed.select('cluster', 'topicDistribution').collect())
    cluster_topicTerms = []
    for e in cluster_topicDist:
        m = list(e[1]).index(max(e[1]))
        cluster_topicTerms.append(tops[m])

    df2_ = []
    for k in sorted(communities.keys()):
        df2_.append((k, ' - '.join(communities[k]), ' - '.join(cluster_topicTerms[k])))
    
    partitionsData2 = spark.createDataFrame(df2_, ['cluster', 'page names', 'LDA topics'])
    partitionsData2.select('cluster', 'LDA topics').show(truncate=False)
    return partitionsData2

def create_df(part_cat_dict):
    df_ = []
    for p in part_cat_dict:
        df_.append((p, ' '.join(part_cat_dict[p]).replace('_', ' ').replace(',', '').replace('\\\'', ' ').replace('(', '').replace(')', '').replace('–',' ').lower()))

    partitionsData = spark.createDataFrame(df_, ['cluster', 'categories'])
    return partitionsData

def topics_with_ml(communities, part_cat_dict):
    
    partitionsData = create_df(part_cat_dict)
    tokenized = tokenize(partitionsData)
    cleaned = stop_words_remove(tokenized)    
    lemmatized = lemmatize(cleaned)
    cvModel = cv_fit(lemmatized)
    cv = cv_transform(cvModel, lemmatized)
    rescaled = idf(cv)
    ldaModel = lda_fit(rescaled)
    ldaTransformed = lda_transform(ldaModel, rescaled)
    tops = show_topic_description(ldaModel, cvModel)
    final = get_topics(communities, tops, ldaTransformed)
    return final

def betweenness_centrality_nodes(graph, clusters_dict):
    res = {}
    for e in clusters_dict:
        H = graph.subgraph(clusters_dict[e])
        d = nx.algorithms.centrality.betweenness_centrality_subset(H, H.nodes, H.nodes)
        
        m = 0
        n = None
        for i in d:
            if d[i] > m:
                m = d[i]
                n = i
        res.setdefault(e, n)
    
    return res

def max_pagerank_on_clusters(dir_graph, clusters_dict):
    res = {}
    for e in clusters_dict:
        H = dir_graph.subgraph(clusters_dict[e])
        pr = nx.algorithms.link_analysis.pagerank_alg.pagerank(H)
        m = 0
        n = None
        for i in clusters_dict[e]:
            if pr[i] > m:
                m = pr[i]
                n = i
        res.setdefault(e, n)
    
    return res

def max_degree_on_clusters(graph, clusters_dict):
    res = {}
    for e in clusters_dict:
        H = graph.subgraph(clusters_dict[e])
        d = H.degree
        m = 0
        n = None
        for p in clusters_dict[e]:
            if d[p] > m:
                m = d[p]
                n = p
        res.setdefault(e, n)
    
    return res

def add_column(df, c, name:str, typ):
    def get_el(i):
        return c[i]
    
    udf_get_el = udf(get_el, typ)
    
    a = df.withColumn(name, udf_get_el('cluster'))
    
    return a

def add_columns(df, l):
    r = df
    for col, tag in l:
        r = add_column(r, col, tag, T.StringType())
    
    return r

def add_attributes_from_df(graph, df):
    rowList = df.collect()
    d = {}
    for row in rowList:
        for p in row[1].split(" - "):
            d.setdefault(p, {"community": row[0], "lda": row[2]})
            
    nx.set_node_attributes(graph, d)
    
    return graph


def main(args):
    os.environ.setdefault('JAVA_HOME', args.jdk8Path)
    user = os.environ.get('USER')
    nltk.download('wordnet')

    global language
    language = 'en'

    mypath = args.inputPath
    max_num_communities = args.nOfClusters
    output_path = args.outputPath


    from os import walk

    (_, _, filenames) = next(walk(mypath))

    global language_mapper
    language_mapper = {
        'en': 'Hidden_categories',
    }

    global driver
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=(args.neo4jUsername, args.neo4jPassword))

    global spark
    spark = SparkSession.builder.appName('graph processing').config("spark.master", "local[*]").config("spark.sql.warehouse.dir", "/home/"+user+"/warehouse").getOrCreate()
    print("Access UI on : " + spark.sparkContext.uiWebUrl)
    print("Spark warehouse set to :", spark.conf.get('spark.sql.warehouse.dir'))

    for f in sorted(filenames):
        path = mypath + f
        (G_undir, G_dir, communities, part_cat_dict) = ld(path, max_num_communities)

        maxx = find_max_freq(count_all_frequencies(part_cat_dict))

        lda_df = topics_with_ml(communities, part_cat_dict)

        betweenness_central_nodes = betweenness_centrality_nodes(G_undir, communities)

        pr_iso_result = max_pagerank_on_clusters(G_dir, communities)

        deg_iso_result = max_degree_on_clusters(G_undir, communities)

        res = add_columns(lda_df,[(betweenness_central_nodes, 'betweenness central node'),
                                  (pr_iso_result, 'max isolated pagerank'),
                                  (deg_iso_result, 'max isolated degree'),
                                  (maxx, 'max category')])    

        print('betweenness central nodes visualization')
        res.select('cluster', 'LDA topics', 'betweenness central node').show(truncate=False)

        print('max pagerank visualization')
        res.select('cluster', 'LDA topics', 'max isolated pagerank').show(truncate=False)

        print('max deg visualization')
        res.select('cluster', 'LDA topics', 'max isolated degree').show(truncate=False)

        G = add_attributes_from_df(G_undir, res)

        name = f[12:-5]
        path = output_path+name+"/"
        #res.coalesce(1).write.option("header", "true").csv(path, mode = 'overwrite')
        res\
        .select("cluster", 'LDA topics', 'betweenness central node', 'max isolated pagerank', 'max isolated degree')\
        .coalesce(1).write.option("header", "true").csv(path, mode = 'overwrite')
        nx.write_gexf(G, path+name+".gexf")

    driver.close()
    spark.stop()
    return True

import argparse

def parseArguments():
    # Create argument parser
    parser = argparse.ArgumentParser()

    # Optional arguments
    parser.add_argument("-jdk8", "--jdk8Path", help="path to jdk8. Default : /usr/lib/jvm/java-1.8.0-openjdk-amd64", type=str, default='/usr/lib/jvm/java-1.8.0-openjdk-amd64')
    parser.add_argument("-n4jadr", "--neo4jAddress", help="neo4j database address. Default : bolt://localhost:7687", type=str, default='bolt://localhost:7687')
    parser.add_argument("-n4jusr", "--neo4jUsername", help="neo4j database username. Default : neo4j", type=str, default='neo4j')
    parser.add_argument("-n4jpwd", "--neo4jPassword", help="neo4j database password. Default : neo4j", type=str, default='neo4j')
    parser.add_argument("-ip", "--inputPath", help="path of the directory containing the graphs. Default : graphs/", type=str, default='graphs/')
    parser.add_argument("-n", "--nOfClusters", help="max number of clusters to extract. Default : 20", type=int, default=20)
    parser.add_argument("-op", "--outputPath", help="path of the output directory. Default : output/", type=str, default='output/')

    # Parse arguments
    args = parser.parse_args()

    return args
    
main(parseArguments())
