# Hybrid4Cast: Cost-effective time series (TS) forecasting with Convolutional Neural Networks (CNN) and Exponential Smoothing (ES)

```Hybrid4Cast``` is a Python package implements a cost-efficient hybrid models integrating ES with CNN for TS forecasting tasks. It contains the entire pipeline including modules like fitting ES models, generating hybrid samples, CNN model optimization, training and forecasting. It also contains the scripts evaluating the performance (ranking the 2nd in terms of point forecast with 4x faster) over the M4 competition. It also contains implementation on Pyspark for medium to large size TS data sets.

Package documentation under construction. Please see examples in hybrid4cast.evaluator for instructions.

## Installation
### Source
From the package directory you then need to install the requirements and then the package (best in a clean virtual environment)
```
$ pip install -r requirements.txt
$ pip install -e .
```

To build the package, use following command:
 python setup.py bdist_egg

### Main Requirements
- [python](http://python.org)
- [tensorflow](https://www.tensorflow.org/)
- [keras](https://keras.io/)
- [scikit-optimize](https://github.com/scikit-optimize/scikit-optimize)

## Examples
- [Evaluation on M4](https://github.com/Aspriter/Hybrid4Cast/evaluator)

## Authors: [Tong Guan](https://github.com/Aspriter), Lei Ma

## Acknowledgements
Special thanks to [Toby Bischoff](http://github.com/bischtob) and Austin Gross for privding us the basic CNN model.
