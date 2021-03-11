%% COMPUTER VISION AND PATTERN RECOGNITION COURSEWORK 2: PATTERN RECOGNITION
%  Patrick McCarthy, pm4617, CID:01353165 & Maria Arranz, ma8816, CID:01250685

%% Section A - Data Preparation

% 1 - Plot time series data

% load data
dir_name = 'PR_CW_DATA_2021';
file_name = 'kitchen_sponge_114_02_HOLD'; % file name - change this to load different files
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

timestep = 612;                                 % sample to use for comparison
PVT = struct;                                   %  struct for data for all materials at chosen timestep
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
colours = linspace(1,1000,length(pressure));
scatter3(pressure,vibration,temperature,20,colours,'filled')
xlabel('pressure')
ylabel('vibration')
zlabel('temperature')
title('Scatter Plot for PVT at timestep 500')

%% Section B - Principal Component Analysis
A = [pressure;vibration;temperature];
C = cov(A);

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
MdlLinearPV = fitcdiscr(PV,names);
K_PV = MdlLinearPV.Coeffs(1,2).Const;  
L_PV = MdlLinearPV.Coeffs(1,2).Linear;

% LDA pressure-temperature
PT = [pressure;temperature]';
MdlLinearPT = fitcdiscr(PT,names);
K_PT = MdlLinearPT.Coeffs(1,2).Const;  
L_PT = MdlLinearPT.Coeffs(1,2).Linear;

% LDA temperature-vibration
TV = [temperature;vibration]';
MdlLinearTV = fitcdiscr(TV,names);
K_TV = MdlLinearTV.Coeffs(1,2).Const;  
L_TV = MdlLinearTV.Coeffs(1,2).Linear;

% plot pressure-vibration
figure
gscatter(PV(:,1),PV(:,2),names)
hold on
xlabel('pressure')
ylabel('vibration')
legend('black foam','car sponge')
f_PV = @(x1,x2) K_PV + L_PV(1)*x1 + L_PV(2)*x2;
fimplicit(f_PV);

% plot pressure-temperature
figure
gscatter(PT(:,1),PT(:,2),names)
hold on
xlabel('pressure')
ylabel('temperature')
legend('black foam','car sponge')
f_PT = @(x1,x2) K_PT + L_PT(1)*x1 + L_PT(2)*x2;
fimplicit(f_PT);

% plot temperature-vibration
figure
gscatter(TV(:,1),TV(:,2),names)
hold on
xlabel('temperature')
ylabel('vibration')
legend('black foam','car sponge')
f_TV = @(x1,x2) K_TV + L_TV(1)*x1 + L_TV(2)*x2;
fimplicit(f_TV);

%% 2 - Apply LDA to 3D data

% LDA pressure-vibration-temperature
PVT = [pressure;vibration;temperature]';
MdlLinearPVT = fitcdiscr(PVT,names);
K_PVT = MdlLinearPVT.Coeffs(1,2).Const;  
L_PVT = MdlLinearPVT.Coeffs(1,2).Linear;

% plot pressure-vibration-temperature
colors = 'rgb';
figure
scatter3(PVT(1:10,1),PVT(1:10,2),PVT(1:10,3),colors(1),'filled')
hold on
scatter3(PVT(11:20,1),PVT(11:20,2),PVT(11:20,3),colors(2),'filled')
grid on
xlabel('pressure')
ylabel('vibration')
zlabel('temperature')
hold on
f_PVT = @(x1,x2) K_PVT + L_PVT(1)*x1 + L_PVT(2)*x2 + 2495;
fsurf(f_PVT,'FaceColor',[0 0 1],'EdgeColor','none');
alpha 0.5

%% Section D - Clustering
