# How to run sample

```
mkdir build
cd build
cmake -S .. \
  -DGGML_BUILD_EXAMPLES=ON \
  -DGGML_BUILD_TESTS=OFF \
  -DGGML_METAL=OFF \
  -DGGML_ACCELERATE=OFF \
  -DGGML_BLAS=OFF \
  -DGGML_OPENMP=OFF

cmake --build . --target single-backend-scheduler -j 4
```
