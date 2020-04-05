# Project basis. 

The aim of the project is using 2006 to 2017 IBM's stock values to predict the stock values of 2018:

![Aim](https://raw.githubusercontent.com/pipegalera/side_projects/master/stock_prices_prediction/figures/ibm_stock_price.png)

I have used 2 types of Recurrent neural networks for this porpuse: ***LSTM and GRU models***.

# Non technical explanation of LTSM and GRU explained in 3 minutes.

Neural Network is one of the main tools in Machine learng to predict outcomes. Tradicionally, Neural Networks do not learn from the previous neuron's values into the next ones. To solve this dynamism problem,  I will use recurrent neural networks (RNN) for the project. RNNs can be thought of as multiple copies of the same network, each passing the nueron value to a successor.

But not that fast. **RNNs are no capable of handling “*long-term dependencies*.”** If the present value depends in a long extent of the long run previous value, RNN tend to not being able to process this long term dependency chain. Therefore, they usually do not do a great job predicting long run time series. **With the task of predicting stocks, the  gap between the relevant information and the point where it is needed is very wide**. How we take into account long-term dependencies? With a LSTM and GRU models. 

Long Short Term Memory networks (**LSTM**) and Gated Recurrent Units (**GRU**) are a special kinds of RNN, capable of learning long-term dependencies. These models are powerful enough to learn the most important past behaviors and understand whether or not those past behaviors are important features in making future predictions. 

# A bit more technical: Why they are better than a traditional RNN?

TBecause they have "gates". LSTM and GRU models are able to keep information of long-term dependencies using filters or gates. In essence, these gates decide how much information to keep of the previous neuron state or values, and how much to drop. This makes the optimization problem or the Neural Network less promp to vanishing or exploding gradient problems.

LSTM operates using:

- ***Input gate***: regulates how much of the new cell state to keep.
- ***Forget gate***: regulates how much of the existing memory to forget.
- ***Output gate***: regulates how much of the cell state should be exposed to the next layers of the network. 

GRU operates using:

- ***Update gate***: decides how much of the candidate activation to use in updating the cell state.
- ***Reset gate***: this gate stands between the previous activation neuron and the next candidate activation neuron, forcing to forget previous state.

Basically, the LSTM unit has separate input and forget gates, while the GRU performs both of these operations together via its reset gate.


# Results of the prediction, measured using the root mean square error.



![LSTM and GRU](https://raw.githubusercontent.com/pipegalera/side_projects/master/LSTM%20to%20predict%20Stock%20Prices/figures/combined_ibm_stock_price_pred.png)
