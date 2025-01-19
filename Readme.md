# A Graph Signal Processing Perspective of Network Multiple Hypothesis Testing with False Discovery Rate Control

## Introduction

This code provides the implementation of the paper "A Graph Signal Processing Perspective of Network Multiple Hypothesis Testing with False Discovery Rate Control". The paper is available [online](https://arxiv.org/abs/2408.03142).

## Implementation

1. Create a virtual environment and install the requirements:
```
conda create --name env_name python=3.9.21
conda activate env_name
cd path/to/this/repo
pip install -r requirements.txt
```

2. Set the working directory to `/scripts`:
```
cd scripts
```

3. To generate the example figures:
```
python comm_data_illus_generate.py # generate the data
python comm_data_illus_performance.py # generate the performance
python example_illus.py # generate the figures after example.
```

4. To produce the results in the paper:
```
python comm_data_fast_fade_generate.py # generate the data
python main.py --method_names MHT-GGSP MHT-GGSP_cens MHT-GGSP_reg lfdr-sMoM BH FDR-smoothing SABHA AdaPT# generate the performance
```
The result figures can be found in the subfolders of the folder `/results`.

## Remark
the folder `/data_for_R` provides the data can be used by R.
You can change the fast fading channel model and parameters in the file 
`scripts/parameters.py` and run the file to simulate the field under different scenarios. After that,
specify the data and results path in `scripts/utils.get_config`.

If you use git commit and allow it to replace LF by CRLF, then you may get errors when trying to load 
.pkl files. You can use the following command to avoid this issue:
```
git config --global core.autocrlf false
```
More details about this issue can be found [here](https://gist.github.com/NateWeiler/df202280ce8cc38e9f00dbc17708fab2).

## Citation
If you use this code in your research, please cite the following papers:

X. Jian, M. Gölz, F. Ji, W. P. Tay, and A. M. Zoubir, “A graph signal
processing perspective of network multiple hypothesis testing with false
discovery rate control,” arXiv preprint arXiv:2408.03142, 2024.

M. Gölz, A. M. Zoubir, and V. Koivunen, “Multiple hypothesis testing
framework for spatial signals,” IEEE Trans. Signal Inf. Process. Netw.,
vol. 8, pp. 771–787, 2022.