# **Transformer and Time Series Data**
This code introduces our research on the use of the Transformer model for time series prediction.

## Introduction
### What is a time series?
First, a time series is a sequence of numerical values representing the evolution of a certain quantity over time (stock market, electricity consumption, battery life, etc.). These series form data sets that can be used due to their numerical values. 
In fact, they are used in the application of numerous mathematical and statistical theories. Their discrete-time temporal dimen-sion is formed with a sampling frequency, e.g. the average temperature over a year.  One thing leading to another, these series find their way into Artificial Intelligence, for analysis and prediction.
When applying these data, it's important to start by analyzing the behavior of the series, with potential patterns of repetition such as seasonality (periodic phenomena that recur throughout the series), trends (the increase or decrease of the data), peaks of intensity or outliers, stationarity...

### What is the Transformer? 
This mechanism allows the model to focus on specific parts of the data, giving each one a varying degree of importance. In this way, it avoids the phenomenon of recurrence, by proceeding with information as a whole rather than one by one, or convo-lution, which tends to take up a lot of memory and execution time. In this way, it creates dependencies between the different variables, enabling the model to be run with greater parallelization; by dividing the Neural Networks into segments, each segment is then processed by a different processor.
The Transformer uses an autoregressive Encoder/Decoder structure, allowing data series to be explained by their past values. The architecture of this model will be explained later in the report.
The Transformer model was created for natural language processing, but its methodology can be extended to the prediction of time series. It is the state of the art for transduction tasks. Its ability to create dependencies between variables makes it one of the most powerful models in machine learning.

### Objective
The aim is to innovate the analysis and prediction of these time series. One effective model is the Transformer.  But this model was created for natural language processing, and is based on the attention mechanism rather than recurrence and convolution. Our aim is therefore to adapt this model to these series in order to optimize prediction.
To do this, we set ourselves specific objectives, such as modeling the data with volatility analysis, potential variable additions and understanding what these time series and their context re-present. This is followed by the precise definition of hyperparameters. For effective comparison, we train traditional models. Then we develop the Transformer model, reducing the prediction error as far as possible, to give an optimized result that's as true as possible.

## **Authors and Acknowledgment**
- **[Arthur Adnot](https://github.com/0sfolt)**
- **[Fabien Chevalier](https://github.com/Lescolopendre)**
- **[Mélodie Desbos](https://github.com/meluxis)**
  
Thank you to all the contributors for their hard work and dedication to the project.

## **Usage**

To use Transformer, follow these steps:

1. Open the project in your favorite code editor.
2. Modify the source code to fit your needs.
3. Build the project: **`npm run build`**
4. Start the project: **`npm start`**
5. Use the project as desired.

## **Contributing**

If you'd like to contribute to Project Title, here are some guidelines:

1. Fork the repository.
2. Create a new branch for your changes.
3. Make your changes.
4. Write tests to cover your changes.
5. Run the tests to ensure they pass.
6. Commit your changes.
7. Push your changes to your forked repository.
8. Submit a pull request.
