#!/usr/bin/env bash
# This script cleans and rebuilds the HNSWLib C++ library and its Python bindings.
# It should be placed and run in the root directory of the cloned HNSWLib repository.

set -e

echo "--- (1/5) Cleaning C++ build directory..."
rm -rf build
mkdir build
cd build

echo "--- (2/5) Building C++ library with CMake & Make..."
cmake ..
make -j

echo "--- (3/5) Cleaning Python binding artifacts..."
cd ../python_bindings/
rm -rf build dist hnswlib.egg-info

echo "--- (4/5) Uninstalling any pre-existing hnswlib package..."
pip uninstall -y hnswlib || true

echo "--- (5/5) Installing Python bindings..."
cd ..
pip install -e .

echo "--- ✅ Rebuild Complete! ---"