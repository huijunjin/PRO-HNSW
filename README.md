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


# PRO-HNSW: Proactive Repair and Optimization for High-Performance Dynamic HNSW Indexes

This repository contains the source code and experimental artifacts for our paper submitted to **[Conference Name, e.g., ICDE 2025]**.

Our framework, PRO-HNSW, introduces a suite of in-place repair modules to enhance the performance and structural integrity of dynamic HNSW graphs, addressing challenges like performance degradation under frequent updates.

---

## ⚙️ Environment & Setup

### 1. Environment
Our experiments were conducted on the following environment:
- **OS**: Ubuntu 20.04.6 LTS
- **CPU**: 12th Gen Intel(R) Core(TM) i7-12700F
- **Memory**: 94GB
- **Python**: 3.11.7

### 2. Setup Instructions
To set up the environment and install the necessary dependencies, please follow these steps.

**Step 1: Clone the repository**
```bash
git clone [Your Anonymous Repository URL]
cd [Your Repository Name]
