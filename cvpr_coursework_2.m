%% COMPUTER VISION AND PATTERN RECOGNITION COURSEWORK 2: PATTERN RECOGNITION
%  Patrick McCarthy, pm4617, CID:01353165 & Maria Arranz, ma8816, CID:01250685

%% Section A - Data Preparation

% 1 - Plot time series data

% load data
dir_name = 'PR_CW_DATA_2021';
file_name = 'car_sponge_101_08_HOLD'; % file name - change this to load different files
load(strcat(dir_name,'/',file_name,'.mat')) 

% plot pressure
figure
hold on
plot(F0pdc)
plot(F1pdc)
legend(['F0';'F1'])
grid on
xlabel('sample')
ylabel('pressure')
title(['Low Frequency Fluid Pressure for ',file_name],'Interpreter', 'none')

% plot vibration
figure
hold on
plot(F0pac(2,:))
plot(F1pac(2,:))
legend(['F0';'F1'])
grid on
xlabel('sample')
ylabel('vibration')
title(['High Frequency Fluid Vibrations for ',file_name],'Interpreter', 'none')

% plot temperature 
figure
hold on
plot(F0tdc)
plot(F1tdc)
legend(['F0';'F1'])
grid on
xlabel('sample')
ylabel('temperature')
title(['Core Temperature for ',file_name],'Interpreter', 'none')

% plot impedance 
figure
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

% 2 - Sample data for one finger

timestep = 500;                                 % sample to use for comparison
PVT = struct;                                  %  struct for data for all materials at chosen timestep
Electrodes = struct;
myDir = dir_name;
myFiles = dir(fullfile(dir_name,'*.mat'));      % get all mat files in struct
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
  save('F0_PVT_400.mat', 'PVT')
  save('F0_Electrodes_400.mat', 'Electrodes')
end


%% 3 - 3D scatter plot

% load data
F0_PVT_500 = load('F0_PVT_400.mat').PVT;
F0_Electrodes_500 = load('F0_Electrodes_400.mat').Electrodes;

% get data into individual variables
for i = 1:60
    pressure(i) = F0_PVT_500(i).pressure;
    vibration(i) = F0_PVT_500(i).vibration;
    temperature(i) = F0_PVT_500(i).temperature;
end

% scatter plot
figure
colours = linspace(1,10,length(pressure));
scatter3(pressure,vibration,temperature,10,colours,'filled')
xlabel('pressure')
ylabel('vibration')
zlabel('temperature')
title('Scatter Plot for PVT at timestep 500')

%% Section B.1 - Principal Component Analysis - PVT
clc
% get the covariance matrix
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
figure
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
figure
colours = linspace(1,10,length(pressure));
scatter(P(:,1),P(:,2),10,colours,'filled')
%hold on
%vbls = {'PC1','PC2'}; % Labels for the variables
%biplot(coeff(:,1:2),'VarLabels',vbls);
%plot([0 0; vector_1(:,1), vector_2(:,1)], [0 0; vector_1(:,2), vector_2(:,2)])
xlabel('PC1')
ylabel('PC2')
title('Proejected onto Principle Components for F0 PVT Data')

% Create projected matrix onto the Principle Components
figure
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

%% Section B.2 - Principal Component Analysis - Electrodes
clc
% Get electrodes data for one finger (F0) - same as for PVT
E = F0Electrodes;

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
figure
plot(eigenvalues, 'b*-', 'MarkerSize',10); % Plot Eigenvalues per Component Number
xlabel('Component Number') 
ylabel('Eigenvalue')
set(gca,'YTick',0:1:12) % Set more ticks in y-axis
title('Variance PCA Scree Plot for F0 Electrode Data')

% Visualise data with first 3 PC
PC3_Ecoeff = Ecoeff(:,3) % Get THIRD column as PC3
PC2_Ecoeff = Ecoeff(:,2) % Get SECOND column as PC2
PC1_Ecoeff = Ecoeff(:,1) % Get FIRST column as PC1
EPC1 = E*PC1_Ecoeff;     % Project data onto PC1 as 1D
EPC2 = E*PC2_Ecoeff;     % Project data onto PC2 as 1D
EPC3 = E*PC3_Ecoeff;     % Project data onto PC3 as 1D

% Create projected matrix onto the Principle Components
figure
hAxes = axes('NextPlot','add',...           % Add subsequent plots to the axes,
             'DataAspectRatio',[1 1 1],...  % match the scaling of each axis,
             'YLim',[0 20],...              % set the y axis limit,
             'Color','white');              % set background color as white
plot(EPC1,5,'k.','MarkerSize',10);   % Plot data set 1 (at y = 5)
plot(EPC2,10,'m.','MarkerSize',10);  % Plot data set 2 (at y = 10)
plot(EPC3,15,'b.','MarkerSize',10);  % Plot data set 3 (at y = 15)
grid on
yticks([5, 10, 15]) % Show only ticks for the 3 PC
yticklabels({'PC1', 'PC2', 'PC3'}) % Replace tick values with PC names
title('1D PCA Plots for F0 Electrodes')

J = [EPC1 EPC2 EPC3]; % Combine projected data onto PC vectors into one data matrix
J_coeff = [PC1_Ecoeff PC2_Ecoeff PC3_Ecoeff] % Combine PC vectors into one data matrix

figure
colours = linspace(1,10,length(EPC1));
scatter3(J(:,1),J(:,2),J(:,3),10,colours,'filled') % Plot the projected data onto first 3 PC
xlabel('PC1')
ylabel('PC2')
zlabel('PC3')
title('Projection for first 3 PC of Electrode Data')

%% Section C - Linear Discriminant Analysis

%% Section D - Clustering
