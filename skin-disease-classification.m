function SkinDisease_Classification_SVM_ResNet50
clc; close all;

% Reproducibility
rng(42,'twister');

%% 1) Select Dataset Root
dataRoot = uigetdir(pwd,'Select dataset root folder');
if dataRoot == 0
    error('Dataset root not selected.');
end

trainDir = fullfile(dataRoot,'train');
valDir   = fullfile(dataRoot,'valid');
testDir  = fullfile(dataRoot,'test');

assert(isfolder(trainDir) && isfolder(valDir) && isfolder(testDir), ...
    'Dataset must contain train, valid, and test folders.');

%% 2) Load Image Datastores
imdsTrain = imageDatastore(trainDir, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
imdsVal   = imageDatastore(valDir, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
imdsTest  = imageDatastore(testDir, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

%% 3) Binary Label Encoding (0: Eczema, 1: Melanoma)
Ytrain = double(imdsTrain.Labels == 'melanoma');
Yval   = double(imdsVal.Labels   == 'melanoma');
Ytest  = double(imdsTest.Labels  == 'melanoma');

fprintf('Training samples   -> Eczema: %d | Melanoma: %d\n',sum(Ytrain==0),sum(Ytrain==1));
fprintf('Validation samples -> Eczema: %d | Melanoma: %d\n',sum(Yval==0),sum(Yval==1));
fprintf('Test samples       -> Eczema: %d | Melanoma: %d\n\n',sum(Ytest==0),sum(Ytest==1));

%% 4) Deep Feature Extraction using ResNet-50
net = resnet50;
inputSize = net.Layers(1).InputSize;
featureLayer = 'avg_pool';

fprintf('Extracting deep features using ResNet-50...\n');

Xtrain = extractDeepFeatures(imdsTrain,net,inputSize,featureLayer);
Xval   = extractDeepFeatures(imdsVal,  net,inputSize,featureLayer);
Xtest  = extractDeepFeatures(imdsTest, net,inputSize,featureLayer);

%% 5) Z-score Normalization (Training statistics only)
mu = mean(Xtrain,1);
sd = std(Xtrain,[],1);
sd(sd==0) = 1;

XtrainZ = (Xtrain - mu) ./ sd;
XvalZ   = (Xval   - mu) ./ sd;
XtestZ  = (Xtest  - mu) ./ sd;

%% 6) Hyperparameter Tuning (Training set only)
boxGrid = logspace(-3,3,13);
cvInner = cvpartition(Ytrain,'KFold',5);

bestC   = boxGrid(1);
bestAcc = -inf;

fprintf('Hyperparameter tuning using 5-fold CV...\n');

for c = 1:numel(boxGrid)
    accFold = zeros(cvInner.NumTestSets,1);
    for k = 1:cvInner.NumTestSets
        tr = training(cvInner,k);
        te = test(cvInner,k);

        mdl = fitcsvm(XtrainZ(tr,:),Ytrain(tr), ...
            'KernelFunction','linear', ...
            'BoxConstraint',boxGrid(c));

        pred = predict(mdl,XtrainZ(te,:));
        accFold(k) = mean(pred == Ytrain(te));
    end

    if mean(accFold) > bestAcc
        bestAcc = mean(accFold);
        bestC   = boxGrid(c);
    end
end

fprintf('Optimal BoxConstraint = %.5g\n\n',bestC);

%% 7) Final Training and Probability Calibration
svmModel = fitcsvm(XtrainZ,Ytrain, ...
    'KernelFunction','linear', ...
    'BoxConstraint',bestC);

svmModel = fitPosterior(svmModel);

%% 8) Performance Evaluation
predTrain = predict(svmModel,XtrainZ);
predVal   = predict(svmModel,XvalZ);
predTest  = predict(svmModel,XtestZ);

accTrain = mean(predTrain == Ytrain);
accVal   = mean(predVal   == Yval);
accTest  = mean(predTest  == Ytest);

fprintf('Training Accuracy   : %.2f %%\n',accTrain*100);
fprintf('Validation Accuracy : %.2f %%\n',accVal*100);
fprintf('Test Accuracy       : %.2f %%\n\n',accTest*100);

%% 9) Advanced Statistical Metrics (Test Set)
cm = confusionmat(Ytest,predTest);
TN = cm(1,1); FP = cm(1,2);
FN = cm(2,1); TP = cm(2,2);

Precision   = TP / max(1,(TP+FP));
Recall      = TP / max(1,(TP+FN));
Specificity = TN / max(1,(TN+FP));
F1          = 2*Precision*Recall / max(1,(Precision+Recall));
BalancedAcc = (Recall + Specificity)/2;

po = (TP+TN)/sum(cm(:));
pe = ((TP+FP)*(TP+FN) + (FN+TN)*(FP+TN)) / (sum(cm(:))^2);
Kappa = (po - pe) / max(eps,(1 - pe));

MCC = (TP*TN - FP*FN) / sqrt(max(eps,(TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)));

[~,scoreTest] = predict(svmModel,XtestZ);
[~,~,~,AUC]   = perfcurve(Ytest,scoreTest(:,2),1);

fprintf('AUC-ROC           : %.4f\n',AUC);
fprintf('Precision         : %.4f\n',Precision);
fprintf('Recall            : %.4f\n',Recall);
fprintf('Specificity       : %.4f\n',Specificity);
fprintf('F1-score          : %.4f\n',F1);
fprintf('Balanced Accuracy : %.4f\n',BalancedAcc);
fprintf('Cohen''s Kappa     : %.4f\n',Kappa);
fprintf('MCC               : %.4f\n\n',MCC);

%% NOTE:
% Trained model weights are intentionally NOT saved or shared.
% This repository provides the implementation methodology only,
% in accordance with academic reproducibility and data privacy policies.

end

%% ================= Helper Function =================
function X = extractDeepFeatures(imds,net,inputSize,layerName)

N = numel(imds.Files);
X = zeros(N,2048,'single');

for i = 1:N
    img = imread(imds.Files{i});
    if size(img,3) == 1
        img = repmat(img,[1 1 3]);
    end
    img = imresize(img,inputSize(1:2));
    X(i,:) = activations(net,img,layerName,'OutputAs','rows');
end

end
