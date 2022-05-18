function [SNet,TNet] = train(SXTrain,SYTrain,TXTrain,TYTrain,sceneTags)
    
    % Function: Train a Seq-Seq LSTM classification network (from 
    %           instructional activities to scenes) and seperate train / 
    %           test set.

    % Usage: [SNet,TNet] = train(SXTrain,TXTrain)

    % Author: Song Yishen @ CIT Lab

    % Input:
    %   SXTrain: Student train set (Activities).
    %   SYTrain: Student train set (Scenes).
    %   TXTrain: Teacher train set (Activities).
    %   TYTrain: Teacher train set (Scenes).
    %   sceneTags: Scene tags array.

    % Output:
    %   SNet: Trained LSTM Net using student sequences.
    %   TNet: Trained LSTM Net using teacher sequences.

    % Train student net
    SNetNumFeatures = size(SXTrain{1,1},1);
    SNetNumHiddenUnits = 200;
    SNetClasses = size(sceneTags,1);
    
    SNetlayers = [ ...
        sequenceInputLayer(SNetNumFeatures)
        lstmLayer(SNetNumHiddenUnits,'OutputMode','sequence')
        fullyConnectedLayer(SNetClasses)
        softmaxLayer
        classificationLayer];
    
    options = trainingOptions('adam', ...
        'MaxEpochs',60, ...
        'GradientThreshold',2, ...
        'Verbose',0, ...
        'Plots','training-progress');
    
    SNet = trainNetwork(SXTrain,SYTrain,SNetlayers,options);

    % Train teacher net

    TNetNumFeatures = size(TXTrain{1,1},1);
    TNetNumHiddenUnits = 200;
    TNetClasses = size(sceneTags,1);
    
    TNetlayers = [ ...
        sequenceInputLayer(TNetNumFeatures)
        lstmLayer(TNetNumHiddenUnits,'OutputMode','sequence')
        fullyConnectedLayer(TNetClasses)
        softmaxLayer
        classificationLayer];
    
    options = trainingOptions('adam', ...
        'MaxEpochs',60, ...
        'GradientThreshold',2, ...
        'Verbose',0, ...
        'Plots','training-progress');
    
    TNet = trainNetwork(TXTrain,TYTrain,TNetlayers,options);
    
end
