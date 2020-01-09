# Wikipedia Graph Processor

Wikipedia Graph Processor is a tool written in Python 3.6 aiming to process wikipedia graphs and extract communities and give a description of their topic.

## Usage
### Pre-requisites

You need:
* an installation of [Apache Spark](https://spark.apache.org/) ([JDK8](https://openjdk.java.net/install/) is required to use Spark propoerly)
* [Python3.6](https://www.python.org/) (or higher)
* Deployed [Neo4j](https://debian.neo4j.org/) database (see [here](https://github.com/epfl-lts2/sparkwiki/tree/master/helpers#3-deploy-the-graph-database) for the details)
* Graph output from the LTS2 [SparkWiki tool](https://github.com/epfl-lts2/sparkwiki)

### Execution

The tool can be run using this command
```
./graph.py --arguments
```
or
```
python3 graph.py --arguments
```
#### Arguments
* `-h`,`--help`: show help message and exit
* `-jdk8`,`--jdk8Path`: path to jdk8. Default: /usr/lib/jvm/java-1.8.0-openjdk-amd64
* `-n4jadr`,`--neo4jAddress`: neo4j database address. Default: bolt://localhost:7687
* `-n4jusr`,`--neo4jUsername`: neo4j database username. Default: neo4j
* `-n4jpwd`,`--neo4jPassword`: neo4j database password. Default: neo4j
* `-ip`,`--inputPath`: path of the directory containing the graphs. Default: graphs/
* `-n`,`--nOfClusters`: max number of clusters to extract. Default: 20
* `-op`,`--outputPath`: path of the output directory. Default: output/

### Results
This tool outputs as many folders as graphs. Each folder contains a *.csv file with the clusters and their respective description and a *.gexf file where each node is assigned a class (cluster) and the class' description. 
