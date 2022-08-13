<img src="../figs/logo.png" align="right" width="28%">

# Installation

## Setup Enviroment
This codebase is tested with `torch==1.10.1` and `torchvision==0.11.2`, with `CUDA 11.3`. In order to successfully reproduce the results reported in our paper, we recommend you to follow the exact same versions. However, similar versions that came out lately should be good as well.

**Step 1: Create Enviroment**
```
conda create -n transvae python=3.7
```
**Step 2: Activate Enviroment**
```
conda activate transvae
```
**Step 3: Install PyTorch**
```
conda install pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -c conda-forge
```
**Step 4: Install Other Packages**
```
conda install matplotlib
conda install progressbar2
conda install scikit-learn
conda install scikit-image
```




## Enviroment Summary

We provide the list of all the packages and their corresponding versions used in this codebase:
```
# Name                    Version          Build  Channel
# -------------------------------------------------------
_libgcc_mutex             0.1                        main  
_openmp_mutex             5.1                       1_gnu  
attrs                     21.4.0             pyhd3eb1b0_0  
blas                      1.0                         mkl  
bottleneck                1.3.4            py37hce1f21e_0  
brotli                    1.0.9                he6710b0_2  
bzip2                     1.0.8                h7f98852_4    conda-forge
ca-certificates           2022.4.26            h06a4308_0  
certifi                   2022.5.18.1      py37h06a4308_0  
cloudpickle               2.0.0              pyhd3eb1b0_0  
cudatoolkit               11.3.1               h2bc3f7f_2  
cycler                    0.11.0             pyhd3eb1b0_0  
cytoolz                   0.11.0           py37h7b6447c_0  
dask-core                 2021.10.0          pyhd3eb1b0_0  
dbus                      1.13.18              hb2f20db_0  
expat                     2.4.4                h295c915_0  
ffmpeg                    4.3                  hf484d3e_0    pytorch
fontconfig                2.13.1               h6c09931_0  
fonttools                 4.25.0             pyhd3eb1b0_0  
freetype                  2.11.0               h70c0345_0  
fsspec                    2022.3.0         py37h06a4308_0  
glib                      2.69.1               h4ff587b_1  
gmp                       6.2.1                h58526e2_0    conda-forge
gnutls                    3.6.13               h85f3911_1    conda-forge
gst-plugins-base          1.14.0               h8213a91_2  
gstreamer                 1.14.0               h28cd5cc_2  
icu                       58.2                 he6710b0_3  
imageio                   2.9.0              pyhd3eb1b0_0  
importlib-metadata        4.11.3           py37h06a4308_0  
importlib_metadata        4.11.3               hd3eb1b0_0  
iniconfig                 1.1.1              pyhd3eb1b0_0  
intel-openmp              2021.4.0          h06a4308_3561  
joblib                    1.1.0              pyhd3eb1b0_0  
jpeg                      9e                   h166bdaf_1    conda-forge
kiwisolver                1.4.2            py37h295c915_0  
lame                      3.100             h7f98852_1001    conda-forge
ld_impl_linux-64          2.38                 h1181459_1  
libffi                    3.3                  he6710b0_2  
libgcc-ng                 11.2.0               h1234567_1  
libgfortran-ng            7.5.0               ha8ba4b0_17  
libgfortran4              7.5.0               ha8ba4b0_17  
libgomp                   11.2.0               h1234567_1  
libiconv                  1.17                 h166bdaf_0    conda-forge
libpng                    1.6.37               h21135ba_2    conda-forge
libstdcxx-ng              11.2.0               h1234567_1  
libtiff                   4.0.10            hc3755c2_1005    conda-forge
libuuid                   1.0.3                h7f8727e_2  
libuv                     1.43.0               h7f98852_0    conda-forge
libxcb                    1.15                 h7f8727e_0  
libxml2                   2.9.14               h74e7548_0  
locket                    1.0.0            py37h06a4308_0  
lz4-c                     1.9.3                h9c3ff4c_1    conda-forge
matplotlib                3.5.1            py37h06a4308_1  
matplotlib-base           3.5.1            py37ha18d171_1  
mkl                       2021.4.0           h06a4308_640  
mkl-service               2.4.0            py37h402132d_0    conda-forge
mkl_fft                   1.3.1            py37h3e078e5_1    conda-forge
mkl_random                1.2.2            py37h219a48f_0    conda-forge
munkres                   1.1.4                      py_0  
ncurses                   6.3                  h7f8727e_2  
nettle                    3.6                  he412f7d_0    conda-forge
networkx                  2.6.3              pyhd3eb1b0_0  
numexpr                   2.8.1            py37h6abb31d_0  
numpy                     1.21.5           py37h6c91a56_3  
numpy-base                1.21.5           py37ha15fc14_3  
olefile                   0.46               pyh9f0ad1d_1    conda-forge
openh264                  2.1.1                h780b84a_0    conda-forge
openssl                   1.1.1o               h7f8727e_0  
packaging                 21.3               pyhd3eb1b0_0  
pandas                    1.3.5            py37h8c16a72_0  
partd                     1.2.0              pyhd3eb1b0_1  
pcre                      8.45                 h295c915_0  
pillow                    6.2.1            py37h6b7be26_0    conda-forge
pip                       21.2.2           py37h06a4308_0  
pluggy                    1.0.0            py37h06a4308_1  
progressbar2              3.37.1           py37h06a4308_0  
py                        1.11.0             pyhd3eb1b0_0  
pyparsing                 3.0.4              pyhd3eb1b0_0  
pyqt                      5.9.2            py37h05f1152_2  
pytest                    7.1.2            py37h06a4308_0  
pytest-runner             5.3.1              pyhd3eb1b0_0  
python                    3.7.13               h12debd9_0  
python-dateutil           2.8.2              pyhd3eb1b0_0  
python-utils              2.5.6            py37h06a4308_0  
python_abi                3.7                     2_cp37m    conda-forge
pytorch                   1.10.1          py3.7_cuda11.3_cudnn8.2.0_0    pytorch
pytorch-mutex             1.0                        cuda    pytorch
pytz                      2022.1           py37h06a4308_0  
pywavelets                1.3.0            py37h7f8727e_0  
pyyaml                    6.0              py37h7f8727e_1  
qt                        5.9.7                h5867ecd_1  
readline                  8.1.2                h7f8727e_1  
scikit-image              0.19.2           py37h51133e4_0  
scikit-learn              1.0.2            py37h51133e4_1  
scipy                     1.7.3            py37hc147768_0  
setuptools                61.2.0           py37h06a4308_0  
sip                       4.19.8           py37hf484d3e_0  
six                       1.16.0             pyh6c4a22f_0    conda-forge
sqlite                    3.38.3               hc218d9a_0  
threadpoolctl             2.2.0              pyh0d69192_0  
tifffile                  2020.10.1        py37hdd07704_2  
tk                        8.6.12               h1ccaba5_0  
tomli                     1.2.2              pyhd3eb1b0_0  
toolz                     0.11.2             pyhd3eb1b0_0  
torchaudio                0.10.1               py37_cu113    pytorch
torchvision               0.11.2               py37_cu113    pytorch
tornado                   6.1              py37h27cfd23_0  
typing_extensions         4.2.0              pyha770c72_1    conda-forge
wheel                     0.37.1             pyhd3eb1b0_0  
xz                        5.2.5                h7f8727e_1  
yaml                      0.2.5                h7b6447c_0  
zipp                      3.8.0            py37h06a4308_0  
zlib                      1.2.12               h7f8727e_2  
zstd                      1.4.9                ha95c52a_0    conda-forge
```
