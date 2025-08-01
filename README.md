# Data Scientist (Academic Project) – Web Traffic Prediction for Tasty Bytes

## Classification Model Development in R

I took on the role of Data Scientist to solve a key business challenge for the recipe website "Tasty Bytes": replacing the subjective selection of recipes for the homepage with a data-driven approach. The objective was clear and measurable: develop a machine learning model that could correctly identify at least 80% of recipes that would generate "high traffic," in order to maximize engagement, visibility, and, ultimately, subscriptions.

My responsibility spanned the entire project lifecycle. I began with a critical data validation and cleansing phase. This involved not only the imputation of missing values in nutritional data, but also a key step of logical inference: I interpreted null values (NA) in the target variable high_traffic as "Low Traffic," a necessary and well-founded assumption in the context of the problem to create a functional binary variable. Furthermore, I performed transformations on categorical (category) and text (servings) variables to standardize them.

I then conducted a deep Exploratory Data Analysis (EDA) using ggplot2 to extract insights. I discovered that not all recipe categories were created equal (e.g., 'Drinks' and 'Breakfast' were underperforming, while 'Vegetables' was a solid traffic generator) and that the number of servings was a determining factor, with recipes for 4 people being the most popular.

For model development, I approached the problem as a binary classification task.
First, I established a Linear Regression model as a baseline. Then, to capture the complexity and outliers, I developed a more advanced model: a Generalized Linear Model with Elastic Net Regularization (GLMNET). I used repeated cross-validation to optimize the hyperparameters and selected the final model based on the ROC metric.
The end result was a resounding success. I translated the client's objective ("identify 80% of high-traffic recipes") into the technical metric of Sensitivity (Recall), and the final GLMNET model achieved 85% Sensitivity on the test set, exceeding the goal. Based on historical data indicating that a popular recipe can increase site traffic by 40%, I projected that implementing this model could generate an estimated 34% increase in overall site traffic (85% Sensitivity × 40% impact).

Measurable Achievements:
+ Exceeded the business objective by achieving 85% sensitivity in identifying high-traffic recipes, surpassing the 80% requirement.
+ Projected a potential 34% increase in site traffic, translating the model's performance into a clear, high-impact business KPI.
+ Performed extensive data cleaning and validation, highlighting the ability to make logical decisions (interpreting NAs in the target variable) to resolve data ambiguity.
+ Identified key drivers of recipe popularity through exploratory analysis, such as food category and number of servings.
+ Developed, trained, and compared two machine learning models (Baseline and GLMNET), demonstrating a methodical approach and the ability to select the appropriate algorithm for the complexity of the problem.
+ Successfully translated a business requirement into a key technical metric (Sensitivity),
demonstrating strong alignment between data science and business objectives.
