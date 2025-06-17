# PRO-HNSW: Proactive Repair and Optimization for High-Performance Dynamic HNSW Indexes

This repository contains the source code and experimental artifacts for our paper. 

Our experiments were conducted on python version of **3.11.7**.

## Environment & Setup

### 1. Install requirements
```
pip install -r requirements.txt
```

### 2. Clone original hnswlib code
```
git clone https://github.com/nmslib/hnswlib.git
```

### 3. Overwrite with PRO-HNSW source code



Code replacement \
Overwrite ```~/hnswlib/hnswlib/hnswalg.h``` with ```/src/hnswalg.h``` \
Overwirte ```~/hnswlib/python_bindings/bindings.cpp``` with ```/src/bindings.cpp``` 

Experiment

## Dataset Preparation

Datasets are downloaded in \
https://github.com/erikbern/ann-benchmarks \
Only using **DeepImage, GIST1M, SIFT1M, NYTimes, MNIST, Fashion-MNIST, COCO-I2I, GloVe-25**
download in the 'data'

## Run Experiments
