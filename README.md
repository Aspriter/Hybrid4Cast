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

## Example Code of Evaluation on M4
### Prepare the data
Put the training and testing data of [M4 computation](https://github.com/M4Competition/M4-methods/tree/master/Dataset) into data_root/Train/ and data_root/Test respectively.

### Get ETS fitted result
Depending on your platform, create a M4Evaluator instance, either locally or on Spark. At the end of this part, a list of [hybrid4Cast.classicts.ETSForecaster] containing the fitted ETS models for each TS will be created at "data_root/ets_output_folder/one_m4_type-etsfit.pkl"
```
# spark initialization
m4_eva = M4EvaluatorSpark(data_root, ets_output_folder, hybrid_output_folder, spark_context=sc)

# find the most fit ETS models for each TS, objects saved to "data_root/ets_output_folder/one_m4_type-etsfit.pkl"
m4_eva.run_pure_ets(one_m4_type, damped=False)
```
Running this on Spark is suggested because runing this on a local machine might take a long time. Also, it is suggested to turn off the damped parameter when searching for the best ETS model, since it saves a lot of time and improvement with damped=True is minimal.

### Run hybrid models
Running this step requires the fitted ETS objects is already created at "data_root/ets_output_folder/one_m4_type-etsfit.pkl". You should run the following code on a local machine because a distributed training process with multiple GPU on Spark is not implemented yet.
```
# local initialization
m4_eva_local = M4EvaluatorLocal(data_root, ets_output_folder, hybrid_output_folder)

# hyper-parameter tuning, it will generate the tuned pickle file at "data_root/hybrid_output_folder/one_m4_type-tune.pkl"
res_gp = m4_eva_local.tune_hyper_parameters(one_m4_type, n_calls=20, n_random_starts=10)
# other alternatives
# res_gp = m4_eva_local.tune_hyper_parameters(one_m4_type, n_calls=20, n_random_starts=10, para_space=your_own_para_space)
# res_gp = m4_eva_local.tune_hyper_parameters(one_m4_type, n_calls=20, all_random=True, absfp_etsfit=customized_absolute_path_to_fitted_ETS_ojbectlist)

# run the evaluation on test data set with the best model 
smape_table, time_cost = m4_eva_local.run_optimal_hybrid(one_m4_type, res_gp=res_gp)
```

## Authors: [Tong Guan](https://github.com/Aspriter), Lei Ma

## Acknowledgements
Special thanks to [Toby Bischoff](http://github.com/bischtob) and Austin Gross for privding us the basic CNN model.
