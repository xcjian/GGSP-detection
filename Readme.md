# A Graph Signal Processing Perspective of Network Multiple Hypothesis Testing with False Discovery Rate Control

## Introduction

This code provides the implementation of the paper "A Graph Signal Processing Perspective of Network Multiple Hypothesis Testing with False Discovery Rate Control" by 
xxx. The paper is available at [arXiv](https://arxiv.org/abs/xxxx.10107).

## Implementation

### Setup virtual environment

1. Create a virtual environment and install the requirements:
```bash
conda create --name env_name python=3.9
```
2. Install the requirements:
```bash
pip install -r requirements.txt
```

## Remark
the folder 'data_for_R' provides the data can be used by R.
You can change the fast fading channel model and parameters in the file 
'scripts/parameters.py' and run the file to simulate the field under different scenarios. After that,
specify the data and results path in scripts/utils.get_config.

If you use git commit and allow it to replace LF by CRLF, then you may get errors when trying to read 
smooth-FDR results. You can use the following command to avoid this issue:
```bash
git config --global core.autocrlf false
```

https://gist.github.com/NateWeiler/df202280ce8cc38e9f00dbc17708fab2