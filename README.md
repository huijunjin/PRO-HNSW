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

### 3. Clone PRO-HNSW code
```
git clone https://github.com/huijunjin/PRO-HNSW.git
```

### 4. Overwrite with PRO-HNSW source code
```
cp ./PRO-HNSW/src/bindings.cpp ./hnswlib/python_bindings/bindings.cpp
cp ./PRO-HNSW/src/hnswalg.h ./hnswlib/hnswlib/hnswalg.h
```

### 5. Compile
```
cp ./PRO-HNSW/build.sh ./hnswlib/
cd hnswlib
chmod +x build.sh
./build.sh
```

## Dataset Preparation

### Download datasets into './PRO-HNSW/data' directory with HDF5 format

https://github.com/erikbern/ann-benchmarks 

Only using **DeepImage, GIST1M, SIFT1M, NYTimes, MNIST, Fashion-MNIST, COCO-I2I, GloVe-25** datasets.

## Run Experiments

### 
