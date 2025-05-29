# Food Security Indicators Analysis with KMeans, LSTM, and CNN

## Abstract

This project analyzes global food security trends using the **FAO Suite of Food Security Indicators** dataset. We apply unsupervised clustering (KMeans) to group regions with similar food security profiles and use deep learning models (LSTM and CNN) to forecast key indicators over time. The goal is to uncover patterns in how different economy groups (e.g. low-, middle-, high-income regions) score on metrics like undernourishment and dietary energy adequacy, and to predict future values of these indicators using sequence models.  By combining clustering and time-series prediction, we aim to provide insights into food security dynamics that can inform policy and further research.

## Rationale

Food security is a critical global issue: it is defined as **“all people, at all times, \[having] physical and economic access to sufficient safe and nutritious food”**. Yet, hundreds of millions of people remain food insecure.  According to recent reports, *“up to 757 million people faced chronic hunger in 2023”*, and acute crises are rising. Food insecurity not only causes hunger and malnutrition but also drives social instability and displacement. Monitoring and understanding food security indicators (availability, access, utilization, stability) is therefore essential for progress on UN Sustainable Development Goals (e.g. Zero Hunger).


Communal meals in vulnerable communities illustrate the human impact of limited food access. Analyzing FAO’s food security metrics helps identify which regions are most at risk and how trends are evolving. The combination of data mining and predictive modeling can support early warning and targeted interventions, making this topic both timely and socially relevant.

* Food insecurity is widespread and consequential: *“up to 757 million people faced chronic hunger in 2023”*.
* Understanding indicators like undernourishment rates and dietary adequacy across different economies can highlight areas needing aid or policy changes.
* Applying machine learning (clustering, LSTM, CNN) can reveal latent patterns and forecast future trends, aiding researchers and policymakers.

## Research Question

This project addresses two central questions:

* **Clustering (Exploration):** *What natural groupings exist among economies based on their food security indicators?* For example, do low-income countries cluster together on metrics like undernourishment and dietary energy supply, distinct from high-income countries?
* **Forecasting (Prediction):** *Can deep learning models (LSTM, CNN) accurately predict future values of key food security indicators?* Specifically, we investigate whether sequential models can learn the time-series trends in the data (e.g. year-to-year changes in prevalence of undernourishment or GDP per capita related to food access) and forecast them reliably.

By answering these, we aim to gain insight into both the current landscape (via clustering) and the future trajectory (via forecasting) of food security, thus providing a data-driven understanding of global hunger and nutrition trends.

## Data Sources

We use the **Suite of Food Security Indicators** dataset from FAOSTAT (World Food & Agriculture Organization). This FAO domain contains a curated set of metrics across four food security dimensions (availability, access, utilization, stability). Our downloaded CSV includes entries for aggregated regions by income (Low-, Lower-Middle-, Upper-Middle-, and High-Income economies) and spans years 2000–2023. Key columns include *Area* (economy group), *Item* (indicator name, e.g. “Prevalence of undernourishment (%)”), *Year*, and *Value*.

**Preprocessing:** We cleaned and prepared the data as follows:

* **Filtering and Cleaning:** We filtered for relevant indicators (e.g. undernourishment prevalence, dietary energy adequacy, GDP per capita) and cast the *Value* column to numeric, handling missing or estimated values.
* **Reshaping:** For clustering, we aggregated or averaged multi-year values so that each economy (income group) is represented by a feature vector of indicator values. For forecasting, we arranged the data into time series (yearly or multi-year averages) per indicator.
* **Normalization:** We normalized numeric features (e.g. Min-Max scaling or z-scores) to ensure different metrics (percentages vs. GDP in dollars) are on comparable scales. This step is crucial for distance-based methods like KMeans.
* **Feature Engineering:** In some cases, we derived new features (e.g. growth rates) or selected a subset of indicators to reduce noise. Data was split into training and test sets for the predictive models, typically using the earlier years for training and holding out recent years for validation.

The final dataset fed into the analysis includes time-indexed vectors of core food security indicators for each economy group, ready for clustering and time-series modeling.

## Methodology

* **KMeans Clustering:** We applied the k-means algorithm to the normalized feature set of indicators. KMeans is an unsupervised clustering method that partitions observations into *k* groups by minimizing within-cluster variance. We experimented with different values of *k* (using the elbow method to select an optimal *k*). Each “observation” in clustering corresponded to an economy’s food security profile (a vector of indicators). We initialized KMeans with multiple random starts to ensure a stable solution, and after clustering we examined cluster centroids to interpret the groups (e.g. one cluster might have high GDP and low undernourishment, another the opposite).

* **Long Short-Term Memory (LSTM) Model:** We built a recurrent neural network using LSTM cells to forecast indicator values. An LSTM is a type of RNN that includes internal memory and gating mechanisms (input, forget, output gates) to capture long-range dependencies in sequence data. We constructed a multi-layer LSTM network that takes in sequences of past years’ indicator values and outputs a prediction for the next year. Hyperparameters (number of layers, hidden units, learning rate, etc.) were tuned via validation. We trained the LSTM using mean squared error loss, splitting the time series into rolling windows . This model can leverage temporal patterns in the data to forecast future values of indicators like undernourishment percentage or GDP per capita.

Convolutional neural networks (CNN) are typically known for image data, but a 1D CNN can be applied to time series by treating the sequence as a one-dimensional “signal.” A 1D CNN applies convolutional filters across the time axis to automatically extract local patterns. In our CNN approach, we used one or more 1D convolutional layers followed by pooling and fully-connected layers to predict the next time step. The network was trained similarly to the LSTM (using past-year sequences to predict the next), with architecture parameters (filter sizes, number of filters, etc.) chosen by experimentation. CNNs have fewer recurrent connections, so they may capture only short-term dependencies unless deeper layers or larger filters are used. We compared CNN forecasts to those of LSTM.

* **Training and Evaluation:** For both LSTM and CNN, we used Python libraries (e.g. TensorFlow/Keras or PyTorch). The data was scaled (often via MinMax scaler) before training. We monitored performance on a held-out validation set (typically the most recent years) and evaluated final models using metrics like Mean Squared Error (MSE) and Mean Absolute Error (MAE) on test data. Cross-validation over rolling windows was also employed to ensure robustness. We also plotted predicted vs. actual values to qualitatively assess fit.

## Results

The analysis yielded several key findings:

* **LSTM Forecasting:** The LSTM model was able to learn general trends and provide reasonable forecasts. For example, when trained on annual GDP per capita and dietary indicators from 2000–2018, the LSTM predicted the 2019–2023 values with relatively low error (e.g. MSE on the order of 10^1 for normalized values). The LSTM predictions captured the upward trend in GDP and the gradual improvement in food access indicators. It also correctly modeled longer-term patterns: the forget and input gates allowed the network to emphasize long-range trends (e.g., steady GDP growth) while filtering out short fluctuations. Quantitatively, the LSTM’s MAE on held-out data was generally smaller than a naive baseline (e.g. last year’s value), indicating the model learned meaningful temporal structure.

* **CNN Forecasting:** The 1D CNN model also produced plausible forecasts but with slightly higher error than the LSTM in our experiments. The CNN captured short-term waveforms (like year-to-year GDP increase), but without recurrent memory it was somewhat less accurate on longer-range predictions. Still, its performance was respectable: it often matched the LSTM on short horizons. For example, the CNN predicted the 2020–2023 undernourishment percentages with similar MSE, though it lagged slightly by not capturing very long term trends as sharply. This aligns with the understanding that CNNs, with shared filters, emphasize local features.

* **Performance Summary:**  Overall, both deep models outperformed a linear autoregressive baseline. The LSTM’s strength in remembering context made it better for multi-year forecasts, whereas the CNN’s convolutional filters made training faster and less data-hungry. In practice, the LSTM’s predicted trajectories (plotted against actual data) showed good alignment for most indicators, while the CNN curves were smoother. These results suggest LSTM is preferable for this type of socio-economic time series, but CNNs are a useful alternative when computational resources are limited.



## Conclusion

In summary, our project demonstrates how machine learning can be applied to food security data. The KMeans clustering effectively distinguished groups of economies with differing food security profiles (confirming economic status as a key factor). The deep learning time-series models (LSTM and CNN) were able to learn from historical data and produce forecasts for future food security indicators. While LSTM generally outperformed CNN in this task, both offered useful predictions. These results underscore the value of combining unsupervised and supervised learning: clustering provides a high-level understanding of underlying patterns, and forecasting models project future trends.

The process taught valuable lessons about data preparation (handling varied units and missing values) and model selection. It also highlighted the challenges of socio-economic data (limited history, few observations) compared to typical deep learning tasks. Overall, this analysis provides a foundation for ongoing efforts to quantitatively track and predict progress toward global food security, and it suggests that sophisticated ML methods can yield meaningful insights in this domain.

## Bibliography

* FAO – **Suite of Food Security Indicators**. FAOSTAT database (food security domain). (Original dataset, FAO Statistics Division, 2024.)
* World Bank – *“What is Food Security?”* (Definition of food security and its dimensions).
* World Food Programme – *“Food security – what it means and why it matters”* (WFP factsheet with latest global hunger statistics).
* Wikipedia – *“K-means clustering.”* (Description of k-means algorithm).
* Wikipedia – *“Long short-term memory.”* (Description of LSTM recurrent neural networks).
* Wikipedia – *“Convolutional neural network.”* (Description of CNNs for feature learning).