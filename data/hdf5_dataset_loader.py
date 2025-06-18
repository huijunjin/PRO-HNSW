#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HDF5 dataset loader for datasets from the ann-benchmarks suite.

This module supports loading datasets for L2 (Euclidean), Cosine (Angular),
and Inner Product (IP) spaces.
"""

import os
import h5py
import numpy as np
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Maps the dataset key used in experiment scripts to the actual HDF5 filename.
# The user must ensure these HDF5 files exist in the specified data root path.
DATASET_MAPPING = {
    # Euclidean (L2)
    "sift-128-euclidean": "sift-128-euclidean.hdf5",
    "gist-960-euclidean": "gist-960-euclidean.hdf5",
    "mnist-784-euclidean": "mnist-784-euclidean.hdf5",
    "fashion-mnist-784-euclidean": "fashion-mnist-784-euclidean.hdf5",

    # Angular (Cosine)
    "deep-image-96-angular": "deep-image-96-angular.hdf5",
    "glove-25-angular": "glove-25-angular.hdf5",
    "glove-50-angular": "glove-50-angular.hdf5",
    "glove-100-angular": "glove-100-angular.hdf5",
    "glove-200-angular": "glove-200-angular.hdf5",
    "nytimes-256-angular": "nytimes-256-angular.hdf5",
    "coco-i2i-512-angular": "coco-i2i-512-angular.hdf5",
    "coco-t2i-512-angular": "coco-t2i-512-angular.hdf5",
}

def _load_single_hdf5_dataset(hdf5_full_path: str) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """
    Loads base vectors, query vectors, and ground truth data from a single HDF5 file.

    It assumes the standard ann-benchmarks keys: 'train', 'test', 'neighbors'.

    Args:
        hdf5_full_path (str): The full path to the HDF5 file.

    Returns:
        tuple[np.ndarray, np.ndarray | None, np.ndarray | None]: A tuple containing:
            - base_vectors (np.ndarray): The training/base data.
            - query_vectors (np.ndarray | None): The test/query data, or None if not found.
            - ground_truth_indices (np.ndarray | None): The ground truth neighbors, or None if not found.

    Raises:
        FileNotFoundError: If the HDF5 file does not exist.
        KeyError: If the 'train' key is missing from the HDF5 file.
    """
    if not os.path.isfile(hdf5_full_path):
        raise FileNotFoundError(f"HDF5 file not found at '{hdf5_full_path}'")

    logging.info(f"Loading HDF5 dataset: {os.path.basename(hdf5_full_path)}")
    
    base_vectors, query_vectors, ground_truth_indices = None, None, None

    try:
        with h5py.File(hdf5_full_path, 'r') as f:
            if 'train' in f:
                base_vectors = np.array(f['train'], dtype=np.float32)
                logging.info(f"  - Loaded 'train' vectors: {base_vectors.shape}, dtype: {base_vectors.dtype}")
            else:
                raise KeyError(f"'train' key (base vectors) not found in HDF5 file '{hdf5_full_path}'")

            if 'test' in f:
                query_vectors = np.array(f['test'], dtype=np.float32)
                logging.info(f"  - Loaded 'test' vectors: {query_vectors.shape}, dtype: {query_vectors.dtype}")
            else:
                logging.warning(f"'test' key (query vectors) not found in HDF5 file. Query vectors will be None.")

            if 'neighbors' in f:
                ground_truth_indices = np.array(f['neighbors'], dtype=np.int32)
                logging.info(f"  - Loaded 'neighbors' (ground truth): {ground_truth_indices.shape}, dtype: {ground_truth_indices.dtype}")
            else:
                logging.warning(f"'neighbors' key (ground truth) not found in HDF5 file. Ground truth will be None.")
        
        logging.info(f"Successfully loaded HDF5 file: {os.path.basename(hdf5_full_path)}")
        return base_vectors, query_vectors, ground_truth_indices
    except Exception as e:
        logging.error(f"A critical error occurred while processing HDF5 file '{hdf5_full_path}': {e}")
        raise

def load_dataset(dataset_key_name: str, hdf5_data_root_path: str) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """
    Loads a specific dataset by its key name.

    This function finds the corresponding HDF5 filename from the DATASET_MAPPING
    and loads the data.

    Args:
        dataset_key_name (str): The key for the dataset (e.g., 'sift-128-euclidean').
        hdf5_data_root_path (str): The root directory path where HDF5 files are stored.

    Returns:
        tuple[np.ndarray, np.ndarray | None, np.ndarray | None]: A tuple containing
            base vectors, query vectors, and ground truth indices.

    Raises:
        ValueError: If the provided dataset_key_name is not defined in DATASET_MAPPING.
    """
    dataset_key_name_lower = dataset_key_name.lower()
    if dataset_key_name_lower not in DATASET_MAPPING:
        available_keys = ", ".join(DATASET_MAPPING.keys())
        raise ValueError(
            f"Unknown dataset key: '{dataset_key_name}'. "
            f"Defined keys are: [{available_keys}]"
        )

    hdf5_filename = DATASET_MAPPING[dataset_key_name_lower]
    full_hdf5_path = os.path.join(hdf5_data_root_path, hdf5_filename)
    
    return _load_single_hdf5_dataset(full_hdf5_path)

# Example usage:
if __name__ == '__main__':
    # This block will only run when the script is executed directly
    # It serves as a simple test case.
    # Create a dummy data directory and a dummy HDF5 file for testing.
    
    DUMMY_DATA_DIR = "./dummy_data"
    DUMMY_HDF5_PATH = os.path.join(DUMMY_DATA_DIR, "sift-128-euclidean.hdf5")
    
    if not os.path.exists(DUMMY_HDF5_PATH):
        print("\nCreating a dummy HDF5 file for testing...")
        os.makedirs(DUMMY_DATA_DIR, exist_ok=True)
        with h5py.File(DUMMY_HDF5_PATH, 'w') as f:
            f.create_dataset('train', data=np.random.rand(1000, 128).astype(np.float32))
            f.create_dataset('test', data=np.random.rand(100, 128).astype(np.float32))
            f.create_dataset('neighbors', data=np.random.randint(0, 1000, size=(100, 10)).astype(np.int32))
        print("Dummy file created.")

    print("\n--- Testing dataset_loader.py ---")
    try:
        base, query, gt = load_dataset('sift-128-euclidean', DUMMY_DATA_DIR)
        print("\nTest successful!")
        print(f"Base shape: {base.shape}, Query shape: {query.shape}, GT shape: {gt.shape}")
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"\nTest failed as expected for a missing file or key: {e}")