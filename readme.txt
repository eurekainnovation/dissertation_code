
#######ReadMe#######

The code was developed using ipython notebook. There are two main folders, finalVersion which has the files used for the main experiment, and finalVersion_OneCompany which makes several tests using only one company.
Initial exploration tests are archived in the initial_experiments folder.
There are four main files per experiment:
+ predictions_automation_script:
Extracts the list of companies to test from an excel file and sets the testing process in motion.
+ generate:
Defines the parameters for the test, such as training time frame, calls the function to generate the models, and then calls the function to make the predictions. In the initial tests this was also where the Gaussian KDE functions were generated.
+ gmodel:
Creates the model for the company and stores it in a file for future use.
+ makepredictions:
This is where predictions are made and then the results stored in csv files.
There is also one file to evaluate results:
+ evaluate_predictions:
Extracts simulation results and analyses them.
If predictions generation is run has to have access to internet to download stock quotes from Yahoo Finance, firewalls can cause issues.
Prediction_Automation_Script be run in parallel to speed-up the generation of predictions.

Note: Uses common Python libraries, Matplotlib, Numpy, Scipy, Pandas, Datetime, Time, etc.
