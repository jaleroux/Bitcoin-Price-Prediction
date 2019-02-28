# Bitcoin Hourly Price Movement Prediction
A short foray into world of Bitcoin price predictions.

Here is an overview of my solution to the Glassnode Crypto Challenge.

The solution contains a script file that executes the stages in data preparation, some feature engineering and finally a little model head-to-head. Additionally, there is the application/deployment of the model which is a basic api output of the current price-direction prediction. This is in the /glassnode_demo folder also containerised in docker hub. 

I have opted to only look at the 1-hour prediction model. This is because of time constraints, but I provide an outline of how I would approach it in further work. I have also left out some hyper-parameter optimisation and kept cross validation to a minimum, again because of time constraints. The code is in the following files.

- Script_glassnode.py : to be executed line-by-line
- Functions_glassnode.py : data extraction, feature design, model build functions.
- application_glassnode.py : python api set up 

## (1)  Model and Theory
### BTC-USD Time-Series

I’ve chosen to build a classification model that predicts the 1-hourly price-movement of the BTC-USD series which was trained and tested on the last 2 years of hourly data. The BTC-USD price data was fetched from the ‘cryptocompare’ website through a batch call to their api. 

The dependent, binary variableis is the hourly close-price direction (compared to last period's price) coded as 1 = UP, 0 = DOWN. The features are a combination of (i) previous observations incl. the prices (close,open,high,low value) and volumes (low,high) at t - 1; and (ii) target encoded variables, which are time variables encoded as having a particular mean price corersponding to a certain time-period (e.g. hour, day, month)...

note: for a multi-hour forecast model, the target variable would still be the close price direction change at time t, but the feature variables would be the lagged observations of price changes, starting from lag(x) onwards (lags smaller than x-hours would be disgrarded as this information would not available in an x-hour lookahead forecast model) e.g. x-hour look-ahead  $model(x)$ is predicting target $Y_t$ with features $X_{t-x:maxlag}$ where X are the differenced prices and volumes from lag x to maxlag. 

To ensure that the properties of the price/vol features series in the modelling process are stationary (have a constant mean and variance) I have differenced the series' by their 1st lag. Performing the AugmentedDickeyFuller test on the before-and-after series we can see that stationarity has been achieved (p-value = 0; rejecting the null-hypothesis that a unit root is present). We could also perform more lagging of the features to remove potential autocorrelations.

### Feature transform & design

The function `feature_creation(...)` prepares the feature set for the problem, parameterised by lag start and end. A matrix of price/vol-differenced features $[X_{t-lagstart},...,X_{t-lagend}]$ is firstly created. I am consdering only 1 lag per feature in this experiment, but further experiemtation could allow me to explore a series of lagged features. The target-encoded time features are also created in this function.

We finally implement some min/max scaling to combat against potential high-magnitude/high-range feature values that would disproportionately impact the fitting. It will also ensure that the features are of a scale the activation functions of the LSTM expect to receive.

### Models
3 contender models are built. A logistic regression, ridge_classification and a long-short-term-memory (LSTM) neural network. The basline model for comparison is the persistence model. $\hat(Y)_{t} = Y_{t-1}$ i.e. previous time-step value predicts current time-step.

The f1-score was selected to asses performance. This strikes a balance between precision and recall measures. The function `models_results(...)`

| Model               | Train F1 Score   | Test F1 Score     |
|---------------------|:----------------:|:-----------------:|
|Persist              |         0.41     |       0.41        |
|Logistic Regression  |         0.47     |       0.45        |
|Ridge Regression|    |        0.47      |       0.45        |
|LSTM                 |         0.55     |          0.54     |

1. Logistic Regression is a go-to for classification models and that is why I've begun with it. The main drawbacks are potential overfitting and not being able to capture non-linearities that could exists between features. 5-fold time-series cross validation was used on the training set. 

2. Ridge Classification is also attempted (maybe regularisation can help performance) but no gain in perforance is shown.

3. LSTM model is decided upon as the model of choice within the neural-net framework. The design of this type of recurrent neural network allows for both long and short term information to be represeneted in the network (held within an internal state variable). In the function, `keras_lstm(...)` one can see the network set-up I have chosen and modelling/testing approach. 
The network is set-up to accept that at each time step in the sequence there is 1 sample, with one timestep within it and one feature set. The internal state of the network is maintained throughout the sequence fed to it. (Stateful=True in the Keras `LSTM()` function). However, when we train we must reset the internal state after each epoch is complete, to ensure the next round start with no bias. The batch size I have chosen is 1 observation per batch. This aims to mimic the 'online learning' and one-step forecasting-framework. After each batch the algorithm updates the weights of the network. After 1 epoch the full set of observations have been fed through the network and the state is reset and the weight optmisation begins again.  

Note: For further development, I will need to run this training on a cloud or GPU machine. However, even with minimal training epochs, the f1_score outperforms the 3 other models.

## (2) API and docker set-up
To run api, type:
docker login: username / password
docker run -p 5000:5000 jlerouxdocker/glassnode_demo

Copy the url into browser. 
The next hour’s closing price movement is printed to screen. (Sorry, no json output for this stage)

I chose to use flask to host the api. Since I’m only performing a 1-step look ahead forecast I’m providing no additional arguments, but for a multi-step forecast we could use the GET commands for the hour forecast of interest to the user. 

## (3) Further Work
- increased training epochs on GPU or uploading job to a cloud service like GCP.
- time-series cross validation on the lstm model to tune the hyper-parameters.
- feature selection / correlation tests. do some more traditional time-series analysis to learn more about autocorrelations and reduce spurious feature selction.
- the multi-step forecast. I could generalise this lstm model into a new custom class and build a series of 1,2,3..,n-step look-ahead models.
- enrich feature set with public news sentiment prior to observation. maybe introduce a classifier that scans the sentiement of bitcoin/crypto news site.
- add function testing framework to ensure code reliability and functionality.
    
