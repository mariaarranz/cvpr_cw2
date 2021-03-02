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

%% Section B - Principal Component Analysis

%% Section C - Linear Discriminant Analysis

%% Section D - Clustering
