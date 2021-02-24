# Computer Vision and Pattern Recognition: Coursework 2

## Section A: Data Preparation
1. Use the plot command to view the time series sensor data for the variables Pressure, Vibration and Temperature (PVT) and the Electrodes. Do this for several objects and trials and then choose a single time step that looks like it will allow differentiation between the data for different objects. Explain why you chose that value. Include an example of your data visualisation for one or two object trials in your report.

2. For one finger (F0 or F1), sample the Pressure, Vibration, Temperature time series data into scaler values measured at the time instance (of your selected time step) for each object / trial. Save the data structures together as a .mat file called F0_PVT.mat or F1_PVT.mat. Repeat for the Electrodes data, saving that as another .mat file. Note that all subsequent actions in this coursework will be on the data sets you just created (and therefore only on one of the robot’s fingers).

3. Create a 3D scatter plot of the complete contents of the PVT mat file, with the axis as Pressure, Vibration and Temperature, with different colours used for different objects. Use the same colours for the objects throughout this work.

## Section B: Principal Component Analysis (PCA)
1. Using PCA (Principal Component Analysis) determine the principal components of the PVT data.
      - Report covariance matrix, eigenvalues, and eigenvectors for the data.
      - Replot the Standardised data with the Principal components displayed.
      - Reduce the data to 2-dimensions and replot.
      - Show how the data is distributed across all principal components by plotting as separate 1D number lines.
      - Comment on your findings.
     
2. There are 19 electrodes per sensor, so relationship between the electrodes for different objects cannot be easily visualised as in the last questions.
      - Use PCA to determine the principal components of the electrode data. Report on the variances of each principal components using a Scree plot.
      - Visualize the electrode data using the three principal components with largest variance.
      - Comment on your findings.

## Section C: Linear Discriminant Analysis (LDA)
1. We want to see if we can discriminate two deformable and porous objects by touch: the black foam and the car sponge.
      - Use LDA to split the training data in terms of Pressure vs. Vibration, Pressure vs.Temperature and Temperature vs. Vibration. Plot the results, including a line showing the generated LDA function.
      - Now apply LDA to the three-dimensional PVT data.
      - Comment on the different outcomes. Consider the physical properties of the objects in your answer and how these may have affected the sensor readings.
      - Repeat the LDA analysis with your own choice of two objects. Explain why you have selected those objects for analysis. In other words – what were you trying to test and what did you determine?

## Section D: Clustering
1. Apply your choice of a clustering algorithm (that we covered in class) to the PVT data
      - Visualise the output.
      - Comment on the outcome. Do the clusters correspond to real-life object similarities?
      - Change the distance metric, repeat the clustering and comment on the change in the outcome.
      
2. Now apply the same clustering algorithm to the electrode data that was processed with PCA
from in B.2.b
      - Visualise the output.
      - Apply Bagging (report your parameters) to the (PCA) electrode data and run the clustering algorithm again.
      - Visualise the new output.
      - Comment on your findings.

## Section E: Conclusion
1. Summarise your work.
      - How have the pattern recognition techniques helped us analyse the data?
      - Would you say it is possible to distinguish the objects only using touch?
      - If you wanted to repeat the experiment with a cheaper tactile sensor, what object properties do you think would be most important for it to measure? Justify your answer based on your findings.
      - Our analysis is based on a single time step access all the available data sensor data. Discuss an alternative method we could use to prepare the data for pattern recognition. What are the pros and cons of this other approach?
