# DataSci_FinalProject

This app was created as a tool to analyze gait data from patients with ALS, Parkinson's, Huntington's, or control patients. The data used in this app are stride to stride measures of footfall contact times, derived from readings by force-sensitive resistors that give outputs proportional to the force under each foot. Each patient's data includes time series data of 13 signals (6 each for  the left and right sides): 
1. elapsed time (sec)
2. left stride interval (sec)
3. right stride interval (sec)
4. left swing interval (sec)
5. right swing interval (sec)
6. left swing interval (% of stride)
7. right swing interval (% of stride)
8. left stance interval(sec)
9. right stance interval(sec)
10. left stance interval (% of stride)
11. right stance interval (% of stride)
12. double support interval (sec)
13. double support interval (% of stride)

Users can select either one of the provided text files in the gait_data directory, or can choose their own data. This app graphs the raw and processed data of the chosen waveform signal, and provides a prediction for the disease state of the patient. A random forest model was trained to make these predictions, with a test set accuracy of 0.9375. 

***NOTE
Our model is by no means a method of diagnosis and HAS NOT been scientifically validated. This model has been trained and tested on a small dataset (48 and 16 respectively), and therefore may not work accurately or effectively in a real-world setting. Please consult a professional if you have any concerns related to your gait.


App requirements:
While this flexdashboard was created in R, the data processing and modeling were done through python. Required R libraries include: shiny, reticulate, tidyverse, and plotly. Required python libraries include pandas, scikit-learn, and glob2.

