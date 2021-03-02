---
Authors: "Majid Afshar and Hamid Usefi"
Date: '01/10/20'
---

## Notice
The code of SVFS is available now, and a link to the paper is given. If you need more details and explanation about the algorithm, please contact [Majid Afshar](http://www.cs.mun.ca/~mman23/) or [Hamid Usefi](http://www.math.mun.ca/~usefi/).

Here is a link to the paper : https://www.nature.com/articles/s41598-021-83150-y

## Use case
To determine the most important features using the algorithm described in "Dimensionality Reduction Using Singular Vectors" by Majid Afshar and Hamid Usefi

## Compile
This code can be run using Python 3.2 and above. Also, the following packages should be installed in your environment as the program dependencies:
* Pandas
* Numpy
* Scikit learn
* Networkx

## Run
To run the code, open `main.py` and specify a list of datasets to apply the method. We note the dataset does not have any headers (neither the features nor the samples IDs). You can add any high dimensional dataset to *Datasets* and insert their name in the list of datasets in 'main.py'.


## Datasets
All datasets must be stored in *Datasets* folder. 
As part of our experiments, we use datasets from [Gene Expression Omnibus (GEO)](https://www.ncbi.nlm.nih.gov/geo/), and datasets can be cleaned by this [code](https://github.com/Majid1292/NCBIdataPrep).
