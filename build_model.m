function [layers,options] = build_model(img_shape,num_classes,learn_rate,epochs,optimizer)
layers = [
    imageInputLayer(img_shape)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer([2 2],'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer([2 2],'Stride',2)
    
    fullyConnectedLayer(num_classes)
    softmaxLayer
    classificationLayer
    ];

options = trainingOptions(optimizer, ...
    'InitialLearnRate',learn_rate, ...
    'MaxEpochs',epochs, ...
    'Shuffle','every-epoch', ...
    'Verbose',true, ...                 %Verbose gibt an, ob das Training in der Kommandozeile visualisiert werden soll
    'Plots','training-progress');       %Plot fuer Trainingsprozess wird angezeigt    
end