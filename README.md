# Multi Expression Programming

This is an implmentation of the MEP algorithm defined here:

> Oltean Mihai, D. Dumitrescu, Multi Expression Programming, Technical report, UBB.

Based upon the C++ code [here](https://github.com/mepx/mep-basic-src).

## Running py-mep

Create the conda environment and source it (Linux):

```
conda env create -f environment.yaml
source activate py-mep-dev
```

Example, running with a dataset `python -m mep.main datasets/data1.csv test.py`. This will run a full MEP population evolution to solve the problem specified in the data CSV, determine the best chromosome, prune it, and then convert that chromosome into a functioning python program that can be run by passing in the feature inputs. Example, `python test.py 5 10`.

An example Python program evolved to solve the addition problem of adding together two features (ex: datasets/data1.csv):
```
import sys

if __name__ == "__main__":
    # constants
    constants = [0.45084442258242485, -0.464331279636617, -0.5128830066318446]

    # now the genes
    program = [0] * 3
    program[0] = float(sys.argv[2])
    program[1] = float(sys.argv[1])
    program[2] = program[0] + program[1]


    # print out the final answer
    print(program[2])
```