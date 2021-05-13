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

Users can choose to view the raw and processed version of any of these 12 signals by uploading a file with the correct txt format of columns with time followed by the signals mentioned. New files unrelated to the given data can be uploaded by the user. A final prediction will also be given of what neurodegenerative disease the gait data is predicted to be of. 
