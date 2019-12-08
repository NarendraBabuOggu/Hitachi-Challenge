**1. How did you approach the data pre-processing, data cleaning and data normalization? Attach the Queries used here.** [10 marks]
	As i used python for the entire data analysis and modelling, i used the pandas, numpy libriaries for all the preprocessing and feature engineering.
	Data Cleaning:
		when it comes to data cleaning, the training data has some missing values in cloumns like longitude, latitude etc... 
		I thought of replacing those null values with the data related to near by stations so that the data will have less error when compared with actual data.
		for that, i first implemented the data modeling and divided the data into individual dims and facts and then, i have replaced the null values with near by station details.
	Data Normalization:
		I used a normal approach for data normalization like diving the data into separate groups so that the relevant data will be grouped together like source and destation both represent the same same entity station, so i grouped them together.
		Also, as the data involves time based columns i separated those columns into separate group.
	Data Pre-processing:
		Data pre-processing involves both data cleaning as well as adding any any extra features (Feature Engineering) and scaling the features to fit the model.
		As part of feature engineering i tried different techniques to include extra features by reducing the current features (as the data contains several stations with more trains)
		But, it only some features i found relevant like the probabilities. I created extra columns which contains the propability of particulat train or station or day to be high or low or medium passenger demand.
		it gave me some pretty good results. and when it comes to test_data, i used the same columns and replaced the probabilities with default probability values for the new trains or stations.
		When it comes to realtime, there will be less changes in the station or train details so, i guess this approach will be suited best for this kind of problems.
		Also, i performed feature scaling on mean_halt_times and distances between source and destination stations to fit the model.
		

**2. How did you approach the problem statement? Explain briefly** [5 Marks]
	To understand the problem clearly, i have gone through the entire training and test data sets. the train and test datasets contains details of different
	different trains starting and ending at different locations. To train the ML model we need to find the relation ship between train & test data and target.
	Also, as of my knowledge the passenger demand for a particular train mostly depend on the source station, particular train and not on the date or time at which the train starts.
	but as i started analysing the data i tried with date time features to understand the data, i am unable to find that relation ship. So i thought of implementing the probabilities 
	of the train to high or low medium in demand, and the results are also good, like some trains has high demand though out the data and some has low. later i tried the same analysis 
	with stations as well and found the similar kind of results. i used the same kind of approach for day column as well and tried different ML models like, logistic regression, decisiontree etc..
	finally i found best results with the a gradient boosting model known as "Light Gradient Bossting Model"(LGBM) which me the scores around 44 to 45 on the test data. Also, Logistic regression gave me 
	scores of 40 to 41. As the challenge is limited to basic machine learning models, i didnt applied the deep learning techniques for this problem. 
**3.a. Give an example of similar data available on a larger scale and your recommended Hadoop architecture to maintain the data.** 
	When it comes with harge scale hadoop and spark are playing a major role in the implementation of data engineering pipelines. 
	And, the architecture also highly depends on the requirements of the solution. We can use spark for analysis or processing of data by writing the data pipelines in scala or pyspark.
	if the input data comes in a json or csv format we can handle with normal coding architectures and we can use kafka if the data is a kind of publisher and subscriber model. 
	And we can use the Hive as a data warehouse with out any changes to the current model.
**3.b. What is your recommended maintenance activity for the architecture mentioned above** [5 Marks]
	As per maintanance activity we need to maintain the data warehouse as its size increases with increase in data. and we need to take acre of hadoop clusters if implemented in hadoop ecosystem.
	and as the model is implemented using python we need to update the code to be in sync with later releases.


