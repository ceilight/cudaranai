An **unoptimized** implementation of convolutional neural network made from
scratch (well technically it does have dependencies but they're pre-installed)
in CUDA/C++.

Don't mind the crappy commit history, I just don't have CUDA-enabled hardware
to test locally.

## How to build

GPU hardware that supports CUDA compiler (nvcc) version 12 or above is required.
Other dependencies include Thrust and cuRAND libraries (which are probably
shipped along with the compiler).

```bash
git clone --branch main https://github.com/ceilight/cudaranai
cd cudaranai
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```

## Demo

Table below documents the result of training different classifiers on
[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset after
30 epochs. Refer to this [notebook](https://colab.research.google.com/drive/1PTEictwtufbPmYrmPti-UT2d56daOq6B?usp=sharing)
for full demonstration of the training process.

| Classifier | Parameters | Preprocessing | Optimizer | Run time | Max. test accuracy |
| --- | --- | --- | --- | --- | --- |
| 2 Conv + 1 FC | ~11k | Scaling + Standardization (subtract the mean + divide by stddev) | Adam | 100 s | 0.9063 |
| 2 Conv + 3 FC | ~62k | Scaling + Standardization | SGD | 120 s | 0.9079 |
| 2 Conv + 3 FC | ~62k | Scaling + Standardization | RMSProp | 90 s | 0.9015 |
| 3 Conv + 2 FC | ~193k | None | RMSProp | 180 s | 0.9038 |