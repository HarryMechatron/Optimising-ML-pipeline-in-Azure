# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
This dataset is about a phone call marketing campaign. The original data can be found [@UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing). The dataset can be used (as we are using) to predict if the client will subscribe to a term deposit or not. The target variable is y. 

The best performing run in the first part (HyperDrive run) of the project had an accuracy score of 0.9137 with the hyperparameters:  ['--C', '1.957125789732623', '--max_iter', '500'].

The best performing model in the second part (AutoML) of the project was a VotingEnsemble with an accuracy score of 0.9173. The details will be described below. 

Citation: [Moro et al., 2014] S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, Elsevier, 62:22-31, June 2014

## Scikit-learn Pipeline
**The Architecture**

In the first part of the project, we are using a training script (train.py) that gets the data from the WEB as a TabularDatasetFactory Class object. The goal is to optimize the [LogisticRegression estimator](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) from [scikit learn library](https://scikit-learn.org/stable/index.html) using Azure Machine Learning [HyperDrive package](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive?preserve-view=true&view=azure-ml-py). For this very purpose, the training script also contains a clean_data function that preprocesses the data (TabularDatasetFactory object) and returns two pandas data frames as the predictors and the target.

As described above the dataset is about a phone call marketing campaign. It has 20 predictors (features) and a target. The explanation for the dataset predictors and the target can be found [@UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing).

We are trying the optimize two of the LogisticRegression hyperparameters: 

The first hyperparameter is called 'C' which is a numerical value of float. 'C' is the inverse of regularization strength and must be a positive float. Smaller values specify stronger regularization. The default value for 'C' is 1.0.

The second hyperparameter is called 'max_iter' which is a numerical value of int. 'max_iter' is the maximum number of iterations taken for the solvers to converge. The default value for 'max_iter' is 100.

A typical hyperparameter tuning by using Azure Machine Learning has several [steps](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters). These are:

* Define the parameter search space,
    * We are using [RandomParameterSampling](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.randomparametersampling?view=azure-ml-py) which defines a  random sampling over a hyperparameter search space.
    * We are using [uniform](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.parameter_expressions?view=azure-ml-py#uniform-min-value--max-value-) for obtaining 'C' values that will be used in hyperparameter tuning. Uniform specifies a uniform distribution from which samples are taken.
    * We are using [choice](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.parameter_expressions?view=azure-ml-py#choice--options-) for obtaining 'max_iter' values that will be used in hyperparameter tuning. Choice specifies a discrete set of options to sample from. 
* Specify a primary metric to optimize,
    * We are using 'accuracy' as our primary metric.
* Specify early termination policy for low-performing runs,
    * We are using [BanditPolicy](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.banditpolicy?view=azure-ml-py) with evaluation_interval=4, and slack_factor=0.08
        * Bandit policy defines an early termination policy based on slack criteria, and a frequency and delay interval for evaluation.
        * evaluation_interval is the frequency for applying the policy.
        * slack_factor is the ratio used to calculate the allowed distance from the best performing experiment run.
* Allocate resources,
    * We are allocating a [compute cluster](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-attach-compute-cluster?tabs=python#what-is-a-compute-cluster) with vm_size='STANDARD_D2_V2', and max_nodes=4
* Launch an experiment with the defined configuration
* Visualize the training runs
* Select the best configuration for your model

The HyperDrive is controlled via the udacity-project.ipynb notebook. The flow is described below:

* Initialize the Workspace and the experiment,
* Create a compute cluster with vm_size='STANDARD_D2_V2' and max_nodes=4,
* Specify a parameter sampler,
* Specify a Policy,
* Create a sklearn estimator for use with train.py,
* Create a HyperDriveConfig using the estimator, hyperparameter sampler, and policy,
* Submit the hyperdrive run to the experiment,
* Get the best run and save the model from that run.

**The Parameter sampler**

The parameter sampler defines our search space. We have used the uniform expression for 'C' and choice expression for 'max_iter' as parameter_expressions. parameter_expressions defines *functions* that can be used in HyperDrive to describe a hyperparameter search space. These *functions* are used to specify different types of hyperparameter distributions. The uniform is selected for 'C' because 'C' requires a continuous set of values. the choice is selected for 'max_iter' because 'max_iter' requires a discrete set of values.

**The Early Stopping Policy**

An early termination policy specifies that if you have a certain number of failures, HyperDrive will stop looking for the answer.  As a result, we terminate poorly performing runs with an early termination policy. Early termination improves computational efficiency. We have used evaluation_interval=4 which means that after the fourth interval, the best performing run will be compared with the upcoming runs' scores, and if they are smaller than the best performing run - slack_factor (which is 0.08 for our case) the run will be canceled.

## AutoML
**Description of the AutoML.**

The AutoML run gets the data from the WEB as a TabularDatasetFactory Class object.

The AutoML configuration was as follows:

* experiment_timeout_minutes=30,
    * Each iteration can run 30 minutes before it terminates.
* compute_target=ws.compute_targets['cpu-cluster'],
    * A compute cluster with vm_size='STANDARD_D2_V2' and max_nodes=4.
* task="classification",
    * This is a classification task.
* primary_metric="accuracy",
    * Accuracy is the metric that AutoML will optimize for model selection.
* training_data=train_dataset,
    * train_dataset is the training data to be used within the experiment.
* label_column_name='y',
    * The name of the label(target) column is 'y'
* max_cores_per_iteration=-1,
    *  Use all the possible cores per iteration per child-run.
* max_concurrent_iterations=4, 
    * The maximum number of iterations that would be executed in parallel is 4.
* n_cross_validations=5
    * 5 cross validations to perform when user validation data is not specified.

The AutoML run had 62 iterations. The estimators and the maximum accuracy scores for each estimator of the AutoML run is as follows:

* VotingEnsemble       0.9173
* StackEnsemble        0.9155
* RandomForest         NaN
* XGBoostClassifier    0.9156
* SVM                  NaN
* LogisticRegression   0.9110
* LightGBM             0.9143
* ExtremeRandomTrees   0.9022
* SGD                  0.9086

The best performing model was a VotingEnsemble which was consisted of an xgboostclassifier and a sgdclassifier. The hyperparameters for each estimator are as follows:

* xgboostclassifier -->       base_score=0.5,
                              booster='gbtree',
                              colsample_bylevel=1,
                              colsample_bynode=1,
                              colsample_bytree=1,
                              gamma=0,
                              learning_rate=0.1,
                              max_delta_step=0,
                              max_depth=3,
                              min_child_weight=1,

* sgdclassifier -->           alpha=6.326567346938775,
                              class_weight='balanced',
                              eta0=0.01,
                              fit_intercept=True,
                              l1_ratio=0.26530612244897955,
                              learning_rate='invscaling',
                              loss='log',
                              max_iter=1000,
                              n_jobs=-1,
                              penalty='none',
                              power_t=0.4444444444444444,
                              random_state=None,
                              tol=0.01 

The most important 5 features are:

* Duration,
* nr.employed,
* emp.var.rate
* month
* euribor3m

## Pipeline comparison
**Comparison**

As stated above, the accuracy score of the best run with HyperDrive is 0.9137, the accuracy score of the best run with AutoML is 0.9173. The AutoML score is slightly better than the HyperDrive score. This is because while we are dealing with only one estimator in Hyperdrive, AutoML has a chance to find the best performing estimator. As a result, a better score is more expected with AutoML.

Both of the runs get the data from the WEB as a TabularDatasetFactory Class object.

The HyperDrive uses [LogisticRegression estimator](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) of sklearn library, and AutoML supports several [estimators](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-auto-train#supported-models) described above. The best performing model for AutoML run is a VotingEnsemble which was consisted of an xgboostclassifier and a sgdclassifier.


## Future work
**Improvement for future experiments**

From the AutoML part of the experiment, it is seen that the dataset is imbalanced. Although AutoML seems to handle imbalanced data we can try to handle it manually.

We can use *AUC_weighted* as the primary metric since it is more robust to imbalanced data. 

TensorFlow LinearClassifier and TensorFlow DNN seem to be blocked since they are not [supported](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-auto-train#supported-models) by AutML. We can try some HyperDrive runs with these estimators and see their performance.

Deep learning is disabled. We can include Deep learning and run a new AutoML pipeline.

We can select a smaller size of predictors using the feature importance visualizations generated by the model explainability feature of Azure AutoML and run a new AutoML pipeline. By doing this we may get better performance for our model.

We can try new HyperDrive runs with the best performing estimator (VotingEnsemble) to get a better score.
