%% COMPUTER VISION AND PATTERN RECOGNITION COURSEWORK 2: PATTERN RECOGNITION
%  Patrick McCarthy, pm4617, CID:01353165 & Maria Arranz Fombellida, ma8816, CID:01250685

%% Section A - Data Preparation
clc
% 1 - Plot time series data

% load data
dir_name = 'PR_CW_DATA_2021';
file_name = 'kitchen_sponge_114_02_HOLD'; % file name - change this to load different files
load(strcat(dir_name,'/',file_name,'.mat')) 

% plot pressure
figure(1)
set(gcf,'position',get(0,'ScreenSize'))
subplot(2,2,1)
hold on
plot(F0pdc)
plot(F1pdc)
legend(['F0';'F1'])
grid on
xlabel('sample')
ylabel('pressure')
title(['Low Frequency Fluid Pressure for ',file_name],'Interpreter', 'none')

% plot vibration
subplot(2,2,2)
hold on
plot(F0pac(2,:))
plot(F1pac(2,:))
legend(['F0';'F1'])
grid on
xlabel('sample')
ylabel('vibration')
title(['High Frequency Fluid Vibrations for ',file_name],'Interpreter', 'none')

% plot temperature 
subplot(2,2,3)
hold on
plot(F0tdc)
plot(F1tdc)
legend(['F0';'F1'])
grid on
xlabel('sample')
ylabel('temperature')
title(['Core Temperature for ',file_name],'Interpreter', 'none')

% plot impedance 
subplot(2,2,4)
hold on
for row=[1:19]
    h1 = plot(F0Electrodes(row,:), 'LineStyle','-');
    h2 = plot(F1Electrodes(row,:), 'LineStyle',':');
end
grid on
legend([h1 h2],'F0','F1')
xlabel('sample')
ylabel('impedance')
title(['Electrode Impedance for ',file_name],'Interpreter', 'none')

saveas(figure(1),[pwd '\figures\Section_A_Part_1.jpg']);

% 2 - Sample data for one finger
timestep = 500;                                 % sample to use for comparison
PVT = struct;                                  %  struct for data for all materials at chosen timestep
Electrodes = struct;
myDir = dir_name;
myFiles = dir(fullfile(dir_name,'*.mat'));      % get all mat files in directory
for k = 1:length(myFiles)
  baseFileName = myFiles(k).name;
  fullFileName = fullfile(myDir, baseFileName);
  load(fullFileName);
  % read data from each file into structs - change 'F0' to 'F1' to get data  for different finger
  PVT(k).name = baseFileName;
  PVT(k).pressure = F0pdc(timestep);
  PVT(k).vibration = F0pac(2,timestep);
  PVT(k).temperature = F0tdc(timestep);
  Electrodes(k).name = baseFileName;
  Electrodes(k).impedance = F0Electrodes(:,timestep);
  save('F0_PVT_500.mat', 'PVT')
  save('F0_Electrodes_500.mat', 'Electrodes')
end

%% 3 - 3D scatter plot
clc
% load data
F0_PVT_500 = load('F0_PVT_500.mat').PVT;
F0_Electrodes_500 = load('F0_Electrodes_500.mat').Electrodes;

% get data into individual variables
for i = 1:60
    pressure(i) = F0_PVT_500(i).pressure;
    vibration(i) = F0_PVT_500(i).vibration;
    temperature(i) = F0_PVT_500(i).temperature;
end

% scatter plot
figure(2)
set(gcf,'position',get(0,'ScreenSize'))
colours = linspace(1,10,length(pressure));
scatter3(pressure,vibration,temperature,15,colours,'filled')
xlabel('pressure')
ylabel('vibration')
zlabel('temperature')
title('Scatter Plot for PVT at timestep 500')
saveas(figure(2),[pwd '\figures\Section_A_Part_3.jpg']);

%% Section B.1 - Principal Component Analysis - PVT
clc
% get the covariance matrix
Q = [pressure;vibration;temperature];
Q = Q';

A = [pressure;vibration;temperature];
C = cov(A);                 % C = covaiance matrix

% finding Eigenvectors
[V,D] = eig(C);         % V = the corresponding eigenvectors 
D=diag(D);              % these are the corresponding eigenvalues

% returns the eigenvector for the maximum eigenvalue
maxeigval= V(:,find(D==max(D))); 

% standarize the data
A = bsxfun(@minus,A',mean(A'))./ std(A');

% Do the PCA
[coeff,score,latent] = pca(A); % coeff = feature vector for eigenvectors

% Calculate eigenvalues and eigenvectors of the covariance matrix
SC = cov(A);

[SV,SD] = eig(SC);
SD=diag(SD);

coeff = coeff*-1;   % compared to SV and SD the eigenvector were multiplied by -1 except the last column (unknown why)
coeff(:,end) = coeff(:,end)*-1; % the last eigenvector was as supposed to be so returning it to original

% Re-plot scatter plot for standardized data
figure(3)
set(gcf,'position',get(0,'ScreenSize'))
subplot(1,2,1)
colours = linspace(1,10,length(pressure));
scatter3(A(:,1),A(:,2),A(:,3),10,colours,'filled')
hold on
vbls = {'PC1','PC2','PC3'}; % Labels for the variables
biplot(coeff(:,1:3),'VarLabels',vbls); % plot the PC in the scatter plot
xlabel('Pressure')
ylabel('Vibration')
zlabel('Temperature')
title('Standarized Scatter Plot for F0 PVT Data at timestep 500')
hold off

PC3_coeff = coeff(:,end); %Get last column as PC3
PC2_coeff = coeff(:,2);   %Get SECOND column as PC2
PC1_coeff = coeff(:,1);   %Get FIRST column as PC1

PC1 = A*PC1_coeff;  % Project onto PC1 to get 1D Data
PC2 = A*PC2_coeff;  % Project onto PC2 to get 1D Data
PC3 = A*PC3_coeff;  % Project onto PC3 to get 1D Data

% Create projected matrix onto the Principle Components
coeff(:,end) = []; % Delete last column to get feature vector for 2D data
P = A*coeff;    % Get 2D projected data

% Re-plot scatter plot for standardized data
subplot(1,2,2)
colours = linspace(1,10,length(pressure));
scatter(P(:,1),P(:,2),10,colours,'filled')
axis([-3 3 -3 3])
axis square
xlabel('PC1')
ylabel('PC2')
title('Proejected onto Principle Components F0 PVT Data')
saveas(figure(3),[pwd '\figures\Section_B_Part_1b-c.jpg']);

% Create projected matrix onto the Principle Components
figure(4)
set(gcf,'position',get(0,'ScreenSize'))
hAxes = axes('NextPlot','add',...             % Add subsequent plots to the axes
             'DataAspectRatio',[1 1 1],...    % match the scaling of each axis
             'YLim',[0.5 3.5],...             % set the y axis limit to show all PC
             'Color','white');                % set background color to white
plot(PC1,1,'k.','MarkerSize',10);  %# Plot data set 1 (show at y = 1)
plot(PC2,2,'m.','MarkerSize',10);  %# Plot data set 2 (show at y = 2)
plot(PC3,3,'b.','MarkerSize',10);  %# Plot data set 3 (show at y = 3)
grid on
yticks([1, 2, 3]) % Show only ticks for the 3 PC
yticklabels({'PC1', 'PC2', 'PC3'}) % Replace tick values with PC names
title('1D PCA Plots for F0 PVT Data')
saveas(figure(4),[pwd '\figures\Section_B_Part_1d.jpg']);

%% Section B.2 - Principal Component Analysis - Electrodes
clc
% Get electrodes data for one finger (F0) - same as for PVT
labels = [];
features = [];
for i = [1:60]
    features(:,i) = F0_Electrodes_500(i).impedance;
    labels = [labels, floor((i-1)/10)];
    %E = [E; {F0_Electrodes_500(v).impedance}];
end

E = features;

% standarize the data
E = bsxfun(@minus,E',mean(E'))./ std(E');

% Calculate eigenvalues and eigenvectors of the covariance matrix
SCE = cov(E);
[SVE,SDE] = eig(SCE);
SDE=diag(SDE);
maxeigvalE= SVE(:,find(SDE==max(SDE))); % Get first PC eigenvector

% Get variance of each component by finding corresponding eigenvalues
[Ecoeff Escore eigenvalues] = pca(E); % Check that vatiables match with eigenvectors and eigenvalues found

% Make scree plot of PC 
figure(5)
set(gcf,'position',get(0,'ScreenSize'))
subplot(1,2,1);
plot(eigenvalues, 'b*-', 'MarkerSize',10); % Plot Eigenvalues per Component Number
xlabel('Component Number') 
ylabel('Eigenvalue')
set(gca,'YTick',0:1:15) % Set more ticks in y-axis
title('Variance PCA Scree Plot for F0 Electrode Data')

% Visualise data with first 3 PC
PC3_Ecoeff = Ecoeff(:,3); % Get THIRD column as PC3
PC2_Ecoeff = Ecoeff(:,2); % Get SECOND column as PC2
PC1_Ecoeff = Ecoeff(:,1); % Get FIRST column as PC1
EPC1 = E*PC1_Ecoeff;     % Project data onto PC1 as 1D
EPC2 = E*PC2_Ecoeff;     % Project data onto PC2 as 1D
EPC3 = E*PC3_Ecoeff;     % Project data onto PC3 as 1D

J = [EPC1 EPC2 EPC3]; % Combine projected data onto PC vectors into one data matrix
J_coeff = [PC1_Ecoeff PC2_Ecoeff PC3_Ecoeff]; % Combine PC vectors into one data matrix

electrodes_data = [labels' J];

% Plot the 3D projected Data onto top 3 variance PC
subplot(1,2,2);
colours = linspace(1,10,length(EPC1));
scatter3(J(:,1),J(:,2),J(:,3),10,colours,'filled') % Plot the projected data onto first 3 PC
xlabel('PC1')
ylabel('PC2')
zlabel('PC3')
axis([-6 6 -6 6 -6 6])
axis square
title('Projection for first 3 PC of Electrode Data')
saveas(figure(5),[pwd '\figures\Section_B_Part_2.jpg']);

% Create projected matrix onto the Principle Components 1D
figure(6)
set(gcf,'position',get(0,'ScreenSize'))
hAxes = axes('NextPlot','add',...           % Add subsequent plots to the axes,
             'DataAspectRatio',[1 1 1],...  % match the scaling of each axis,
             'YLim',[0 20],...              % set the y axis limit,
             'Color','white');              % set background color as white
plot(EPC1,5,'k.','MarkerSize',10);   % Plot data set 1 (at y = 5)
plot(EPC2,10,'m.','MarkerSize',10);  % Plot data set 2 (at y = 10)
plot(EPC3,15,'b.','MarkerSize',10);  % Plot data set 3 (at y = 15)
grid on
xlim([-10 10])
yticks([5, 10, 15]) % Show only ticks for the 3 PC
yticklabels({'PC1', 'PC2', 'PC3'}) % Replace tick values with PC names
title('1D PCA Plots for F0 Electrodes')
saveas(figure(6),[pwd '\figures\Section_B_Part_2_Extra.jpg']);

%% Section C - Linear Discriminant Analysis

% isolate data for black foam and car sponge
names = [];
pressure = [];
vibration = [];
temperature = [];

for i = 11:20
    names(i-10) = 0;
    pressure(i-10) = F0_PVT_500(i).pressure;
    vibration(i-10) = F0_PVT_500(i).vibration;
    temperature(i-10) = F0_PVT_500(i).temperature;
end
for i = 21:30
    names(i-10) = 1;
    pressure(i-10) = F0_PVT_500(i).pressure;
    vibration(i-10) = F0_PVT_500(i).vibration;
    temperature(i-10) = F0_PVT_500(i).temperature;
end

% 1 - Split the training data in terms of different pairs of variables, perform LDA and plot

% LDA pressure-vibration
PV = [pressure;vibration]';
PV_std = bsxfun(@minus,PV,mean(PV))./ std(PV); % standardise data
MdlLinearPV = fitcdiscr(PV_std,names);
K_PV = MdlLinearPV.Coeffs(1,2).Const;  
L_PV = MdlLinearPV.Coeffs(1,2).Linear;

% LDA pressure-temperature
PT = [pressure;temperature]';
PT_std = bsxfun(@minus,PT,mean(PT))./ std(PT); % standardise data
MdlLinearPT = fitcdiscr(PT_std,names);
K_PT = MdlLinearPT.Coeffs(1,2).Const;  
L_PT = MdlLinearPT.Coeffs(1,2).Linear;

% LDA temperature-vibration
TV = [temperature;vibration]';
TV_std = bsxfun(@minus,TV,mean(TV))./ std(TV);% standardise data
MdlLinearTV = fitcdiscr(TV_std,names);
K_TV = MdlLinearTV.Coeffs(1,2).Const;  
L_TV = MdlLinearTV.Coeffs(1,2).Linear;

colors = 'rgb';

% plot pressure-vibration
figure(7)
subplot(3,1,1);
set(gcf,'position',get(0,'ScreenSize'))
gscatter(PV_std(:,1),PV_std(:,2),names)
hold on
xlabel('pressure')
ylabel('vibration')
legend('black foam','car sponge','Location','southwest')
title('Scatter Plot with LDA Discriminant Line for Vibration-Pressure')
f_PV = @(x1,x2) K_PV + L_PV(1)*x1 + L_PV(2)*x2;
h1 = fimplicit(f_PV);
set(h1,'Color','k','LineWidth',1,'DisplayName','decision boundary')
grid on

% plot pressure-temperature
subplot(3,1,2);
gscatter(PT_std(:,1),PT_std(:,2),names)
hold on
xlabel('pressure')
ylabel('temperature')
legend('black foam','car sponge','Location','southwest')
title('Scatter Plot with LDA Discriminant Line for Temperature-Pressure')
f_PT = @(x1,x2) K_PT + L_PT(1)*x1 + L_PT(2)*x2;
h2 = fimplicit(f_PT);
set(h2,'Color','k','LineWidth',1,'DisplayName','decision boundary')
grid on
xlim([-2,1.5])
ylim([-5,5])

% plot temperature-vibration
subplot(3,1,3);
gscatter(TV_std(:,1),TV_std(:,2),names)
hold on
xlabel('temperature')
ylabel('vibration')
legend('black foam','car sponge','Location','southwest')
title('Scatter Plot with LDA Discriminant Line for Vibration-Temperature')
f_TV = @(x1,x2) K_TV + L_TV(1)*x1 + L_TV(2)*x2;
h3 = fimplicit(f_TV);
set(h3,'Color','k','LineWidth',1,'DisplayName','decision boundary')
xlim([-2,1.5])
ylim([-5,5])
grid on
saveas(figure(7),[pwd '\figures\Section_C_Part_1a.jpg']);

%% 2 - Apply LDA to 3D data

% LDA pressure-vibration-temperature
PVT = [pressure;vibration;temperature]';
PVT_std = bsxfun(@minus,PVT,mean(PVT))./ std(PVT);% standardise data
MdlLinearPVT = fitcdiscr(PVT_std,names);
K_PVT = MdlLinearPVT.Coeffs(1,2).Const;  
L_PVT = MdlLinearPVT.Coeffs(1,2).Linear;

% plot pressure-vibration-temperature
colors = 'rgb';
figure(8)
set(gcf,'position',get(0,'ScreenSize'))
scatter3(PVT_std(1:10,1),PVT_std(1:10,2),PVT_std(1:10,3),colors(1),'filled')
hold on
scatter3(PVT_std(11:20,1),PVT_std(11:20,2),PVT_std(11:20,3),colors(2),'filled')
grid on
xlabel('pressure')
ylabel('vibration')
zlabel('temperature')
title('Scatter Plot with LDA Discriminant Surface for Pressure-Vibration-Temperature')
hold on
f_PVT = @(x1,x2) K_PVT + L_PVT(1)*x1 + L_PVT(2)*x2;
s1 = fsurf(f_PVT,'FaceColor',[0 0 1],'EdgeColor','none');
set(s1,'DisplayName','decision boundary')
alpha 0.5
grid on
legend('black foam','car sponge','decision surface','Location','northeast')
saveas(figure(8),[pwd '\figures\Section_C_Part_1b.jpg']);

%% Section D - Clustering: K-Means Method
% reference: https://uk.mathworks.com/help/stats/kmeans.html#bue6nc4-1
clc
opts = statset('Display','final');
[idx,I,sum,D] = kmeans(A,3,'Replicates',10,'Options',opts);

% plot K-Means 3D Scatter Plot with centroids
figure(9) 
set(gcf,'position',get(0,'ScreenSize'))
scatter3(A(:,1), A(:,2), A(:,3), 20, idx, 'filled');
xlabel('Pressure')
ylabel('Vibration')
zlabel('Temperature')
hold on
plot(I(:,1),I(:,2),'kx')
title ('Cluster Assignments and Centroids using K-Means')
hold off

%% Section D.1.a - Clustering: Hierarchical Clustering using Eucledian Distance Metric
% reference: https://uk.mathworks.com/help/stats/linkage.html#mw_59e9693d-3784-4a0d-89dd-5dd020a605b2
clc
% For the following code we use the eucledican distance metric (default)
% and the single method for computing the distance (shortest distance - default)

Z = linkage(Q); % original un-standardized data (can be seen in Section B.1)
c = cluster(Z,'Maxclust',6); % create the clustering using 6 clusters 

% Plot dendrogram using euclidean metric
figure(10)
set(gcf,'position',get(0,'ScreenSize'))
subplot(2,2,3);
dendrogram(Z) % create the dendrogram using the linkage of the data
ylabel('Distance')
xlabel('Data Point')
title ('Dendogram of PVT Raw Data using Eucledian Distance Metric')

% Plot clustering scatter plot using Euclidean Metric
subplot(2,2,1);
scatter3(Q(:,1),Q(:,2),Q(:,3),15,c, 'filled') % plot data and separate depending on cluster of linkage
xlabel('Pressure')
ylabel('Vibration')
zlabel('Temperature')
title ('Raw Data Clustering using Euclidean Distance Metric')

% Section D.1.b - Clustering: Hierarchical Clustering using Eucledian Distance Metric
% reference: https://uk.mathworks.com/help/stats/linkage.html#mw_59e9693d-3784-4a0d-89dd-5dd020a605b2
clc

% For the following code we use the manhattan distance metric (cityblock)
% and the single method for computing the distance (shortest distance - default)

Zb = linkage(Q,'single','cityblock'); % original un-standardized data (can be seen in Section B.1)
cb = cluster(Zb,'Maxclust',6); % create the clustering using 6 clusters 

% Plot dendrogram using manhattan metric
subplot(2,2,4);
dendrogram(Zb) % create the dendrogram using the linkage of the data
ylabel('Distance')
xlabel('Data Point')
title ('Dendogram of PVT Raw Data using Manhattan Distance Metric')

% Plot clustering scatter plot using Manhattan Metric
subplot(2,2,2);
scatter3(Q(:,1),Q(:,2),Q(:,3),15,cb, 'filled') % plot data and separate depending on cluster of linkage
xlabel('Pressure')
ylabel('Vibration')
zlabel('Temperature')
title ('Raw Data Clustering using Manhattan Distance Metric')
saveas(figure(10),[pwd '\figures\Section_D_Part_1.jpg']);

%% Section D.2.b - Bagging
clc
% reference: https://uk.mathworks.com/matlabcentral/answers/377839-split-training-data-and-testing-data

% Take the electrode data processed by PCA in Section B.2 (original
% un-standardized data (can be seen in Section B.1))
data = electrodes_data;
% Cross varidation (train: 60%, test: 40%)
cv = cvpartition(size(data,1),'HoldOut',0.4);
idx = cv.test;
% Separate to training and test data
dataTrain = data(~idx,:);
dataTest  = data(idx,:);

classTrain = dataTrain(:,1);
Dtrain = [];
for i=[2:4]
    Dtrain = [Dtrain dataTrain(:,i)];
end

% train the training data
Mdl = TreeBagger(100,Dtrain,classTrain,'OOBPrediction','On','Method','classification','OOBPredictorImportance','off');

view(Mdl.Trees{1},'Mode','graph') % view the first tree generated
view(Mdl.Trees{2},'Mode','graph') % view the second tree generated

% plot the out of bag (oob) error for the model
figure(11)
subplot(1,2,1);
set(gcf,'position',get(0,'ScreenSize'))
oobErrorBaggedEnsembleTrain = oobError(Mdl); % get the oob error of the train model
plot(oobErrorBaggedEnsembleTrain)
grid on
xlabel ('Number of trees')
ylabel ('Out-of-bag classification error')

% Section D.2.c - Confusion Matrix
classTest = dataTest(:,1);
Dtest = [];
for i=[2:4]
    Dtest = [Dtest dataTest(:,i)];
end

predicted_class = []; % created an empty matrix to input the predicted class
for i=1:size(Dtest,1) % loop through the split test data
    predicted = predict(Mdl, Dtest(i,:)); % predict the class using the training model
    predicted = cell2mat(predicted);  % convert cell type into matrix  
    predicted = str2double(predicted); % convert character values into double matrix
    predicted_class = [predicted_class; {predicted}]; % append the predicted values to the predicted_class matrix
end
predicted_class = cell2mat(predicted_class); % convert cell type to matrix type

g1 = classTest;	% Known groups
g2 = predicted_class;	% Predicted groups

% generate the confusion chart/matrix
subplot(1,2,2);
confusionchart(g1,g2)
saveas(figure(11),[pwd '\figures\Section_D_Part_2.jpg']);

