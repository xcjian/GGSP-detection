# A Graph Signal Processing Perspective of Network Multiple Hypothesis Testing with False Discovery Rate Control

## Introduction

This code provides the implementation of the paper "A Graph Signal Processing Perspective of Network Multiple Hypothesis Testing with False Discovery Rate Control" by 
xxx. The paper is available at [arXiv](https://arxiv.org/abs/xxxx.10107).

## Implementation

### Setup virtual environment

1. Create a virtual environment and install the requirements:
```
conda create --name env_name python=3.9.21
conda activate env_name
cd path/to/this/repo
pip install -r requirements.txt
```

3. set the working directory to `/scripts`:
```
cd scripts
```

To generate the example figures:
```
python comm_data_illus_generate.py # generate the data
python comm_data_illus_performance.py # generate the performance
python example_illus.py # generate the figures after example.
```

To produce the results in the paper:
```
python comm_data_fast_fade_generate.py # generate the data
python main.py # generate the performance
```

## Remark
the folder 'data_for_R' provides the data can be used by R.
You can change the fast fading channel model and parameters in the file 
'scripts/parameters.py' and run the file to simulate the field under different scenarios. After that,
specify the data and results path in scripts/utils.get_config.

If you use git commit and allow it to replace LF by CRLF, then you may get errors when trying to load 
.pkl files. You can use the following command to avoid this issue:
```
git config --global core.autocrlf false
```
More details about this issue can be found [here](https://gist.github.com/NateWeiler/df202280ce8cc38e9f00dbc17708fab2).
