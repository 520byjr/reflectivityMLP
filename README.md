# reflectivityMLP

The primary task here is to predict the seismic reflectivity (i.e. the reflection coefficient) based on the impedance contrast across rock interfaces. The provided scripts use several well-log measurements including seismic P and S wave velocities as
well as rock density to determine reflectivity across layer boundaries. Seismic reflectivity describes the partitioning of seismic wave energy at an interface, typically a boundary between two different layers of rock. The reflection coefficient or reflectivity is the proportion of seismic wave amplitude reflected from an interface relative to the amplitude of the incoming wave. If 10% of the amplitude is returned, then the reflection coefficient is 0.10.

The data set for this exercise consists of 500 Vp, Vs, density and reflectivity measurements at different depths within a wellbore. The task is to predict the reflection coefficient for any new data set without knowing the incidence angle of the incoming wave and without any knowledge of the specific functional form that maps Vp, Vs and 𝜚 to reflectivity.

The provided implementation is based on a detailed _SEG, Leading Edge Article_ by Graham Ganssle, Head of Data Science, Expero Inc.
The article can be found here: https://doi.org/10.1190/tle37080616.1
see also: https://github.com/seg/tutorials-2018/blob/master/1808_Neural_networks/Manuscript.md

This implementation uses the **scikit-learn** machine learning package, as well as **numpy** and **matplotlib**


# Workflow

1) Run HW1_dataPreProcess.py to perform the data preprocessing steps, including train/test split and standardization
2) Run HW2_reflectivity.py to train the model and evaluate the model performance on validation and testing data. This implementation used a Multi-Layer-Perceptron for which weights and biases are adjusted iteratively through standard back-projection of errors (see change in cost function with epochs over training data in plots/well_log_learning_curve.png). 

The most basic MLP already achieve good results with R2-scores of about 0.94 which can be increased to 0.99 through hyperparameter tuning.
Additional implementations include LinearRegression() and  SGDRegressor() which also show satifacory performance.
