clear
%Initiale Werte fuer die Funktionen festlegen
img_shape = [28 28 1];
num_classes = 10;
learn_rate = 0.01;
epochs = 4;
optimizer = 'rmsprop';
%Laden des Datensatzes
[xtrain,ytrain,xtest,ytest] = prepare_dataset;
%Build model erstellt das CNN fuer die gegebenen Werte
[layers,options] = build_model(img_shape,num_classes,learn_rate,epochs,optimizer);
% Training
net = trainNetwork(xtrain,ytrain,layers,options);
%Testing
ypred = classify(net,xtest);
%Erstellen einer Konfusionsmatrix fuer die Auswertung
cm = confusionchart(ytest,ypred,'RowSummary','row-normalized','ColumnSummary','column-normalized');
cm.Title = 'MNIST Classification Using CNN';
%Berechnung der Genauigkeit des CNN
accuracy = sum(ypred == ytest)/numel(ytest);

