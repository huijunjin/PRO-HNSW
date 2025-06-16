# PRO-HNSW
This repository contains the source code and experimental artifacts for our paper submitted to "**PRO-HNSW: Proactive Repair and Optimization for High-Performance Dynamic HNSW Indexes**."


Our experiments were conducted on the following environment:

  -**OS**: Ubuntu 20.04.6 LTS \
  -**CPU**: 12th Gen Intel(R) Core(TM) i7-12700F \
  -**Mem**: 94GB \
  -**Python**: 3.11.7 

Datasets are downloaded in \
https://github.com/erikbern/ann-benchmarks \
Only using **DeepImage, GIST1M, SIFT1M, NYTimes, MNIST, Fashion-MNIST, COCO-I2I, GloVe-25**

Requirement are in \
```requirements.txt```

Code replacement \
Overwrite ```~/hnswlib/hnswlib/hnswalg.h``` with ```/src/hnswalg.h``` \
Overwirte ```~/hnswlib/python_bindings/bindings.cpp``` with ```/src/bindings.cpp``` 

Experiment
