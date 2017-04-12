# Multi Expression Programming

This is an implmentation of the MEP algorithm defined here:

> Oltean Mihai, D. Dumitrescu, Multi Expression Programming, Technical report, UBB.

Based upon the C++ code [here](https://github.com/mepx/mep-basic-src).

## Running py-mep

Create the conda environment and source it (Linux):

```
conda env create -f environment.yml
source activate py-mep-dev
```

Example, running with a dataset `python -m mep.main datasets/data1.csv`