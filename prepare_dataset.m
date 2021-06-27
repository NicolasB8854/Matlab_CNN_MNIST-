function [xtrain,ytrain,xtest,ytest] = prepare_dataset
path = addpath(fullfile(matlabroot,'examples','nnet','main'));
filenameImagesTrain = 'train-images.idx3-ubyte';
filenameLabelsTrain = 'train-labels.idx1-ubyte';
filenameImagesTest = 't10k-images.idx3-ubyte';
filenameLabelsTest = 't10k-labels.idx1-ubyte';
xtrain = processImagesMNIST(filenameImagesTrain);   %Trainingsdaten
ytrain = processLabelsMNIST(filenameLabelsTrain);   %Klassenzugehoerigkeiten der Trainingsdaten
xtest = processImagesMNIST(filenameImagesTest);     %Testdaten
ytest = processLabelsMNIST(filenameLabelsTest);     %Klassenzugehoerigkeiten der Testdaten
end