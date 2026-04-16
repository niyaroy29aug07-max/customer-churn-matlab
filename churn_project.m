%% CUSTOMER CHURN PREDICTION PROJECT (DAY 1–6 COMBINED)

clc;
clear;

%% ================= DAY 1: DATA LOADING =================
data = readtable('WA_Fn-UseC_-Telco-Customer-Churn.csv');

disp('First few rows:');
head(data)

disp('Column Names:');
data.Properties.VariableNames

disp('Summary (Before Cleaning):');
summary(data)

disp('Dataset Size:');
size(data)

%% ================= DAY 2: DATA CLEANING =================
data.TotalCharges = str2double(string(data.TotalCharges));
data = rmmissing(data);

data.Churn = categorical(data.Churn);
data.gender = categorical(data.gender);
data.Contract = categorical(data.Contract);

disp('Summary (After Cleaning):');
summary(data)

disp('Cleaned Data Preview:');
head(data)

disp('Cleaned Data Size:');
size(data)

%% ================= DAY 3: DATA VISUALIZATION =================

% Churn Distribution
churnCount = countcats(data.Churn);
figure;
bar(churnCount);
title('Churn Distribution');
xlabel('Churn');
ylabel('Number of Customers');

% Monthly Charges vs Churn
figure;
boxplot(data.MonthlyCharges, data.Churn);
title('Monthly Charges vs Churn');

% Tenure vs Churn
figure;
boxplot(data.tenure, data.Churn);
title('Tenure vs Churn');

%% ================= DAY 4: DATA PREPARATION =================
X = data(:, {'tenure','MonthlyCharges','TotalCharges'});
Y = data.Churn;

cv = cvpartition(height(data), 'HoldOut', 0.2);

XTrain = X(training(cv), :);
YTrain = Y(training(cv));

XTest = X(test(cv), :);
YTest = Y(test(cv));

disp('Training Data Size:');
size(XTrain)

disp('Testing Data Size:');
size(XTest)

%% ================= DAY 5: MODEL BUILDING =================
model = fitctree(XTrain, YTrain);

YPred = predict(model, XTest);

accuracy = sum(YPred == YTest) / numel(YTest);
fprintf('Accuracy: %.2f%%\n', accuracy * 100);

% Decision Tree Visualization
figure;
view(model, 'Mode', 'graph');

%% ================= DAY 6: MODEL EVALUATION =================

% Confusion Matrix
figure;
confusionchart(YTest, YPred);
title('Confusion Matrix');

% Metrics
cm = confusionmat(YTest, YPred);

TN = cm(1,1);
FP = cm(1,2);
FN = cm(2,1);
TP = cm(2,2);

precision = TP / (TP + FP);
recall = TP / (TP + FN);

fprintf('Precision: %.2f\n', precision);
fprintf('Recall: %.2f\n', recall);

