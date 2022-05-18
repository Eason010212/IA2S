function [SNet,TNet,SXTest,SYTest,TXTest,TYTest,sceneTags,classNames] = train(actSeqFile,sceneSeqFile,actTagsFile,sceneTagsFile,trainsetFile)
    
    % Function: train a Seq-Seq LSTM classification network (from 
    %           instructional activities to scenes) and seperate train / 
    %           test set.

    % Usage: [SNet,TNet,SXTest,SYTest,TXTest,TYTest,sceneTags,classNames] = 
    %        train(actSeqFile,sceneSeqFile,actTagsFile,sceneTagsFile,
    %        trainsetFile)

    % Author: Song Yishen @ CIT Lab

    % Input:
    %   actSeqFile: Activity sequence CSV file. For the specific format,
    %               see Readme.md.
    %   sceneSeqFile: Scene sequence CSV file. For the specific format, see
    %                 Readme.md.
    %   actTagsFile: Available activity tags CSV file. For the specific
    %                format, see Readme.md.
    %   sceneTagsFile: Available scene tags CSV file. For the specific
    %                  format, see Readme.md.
    %   trainsetFile: The CSV file that determines the division of the 
    %                 training set from the test set. For the specific
    %                 format, see Readme.md.

    % Output:
    %   SNet: Trained LSTM Net using student sequences.
    %   TNet: Trained LSTM Net using teacher sequences.
    %   SXTest: Student test set (Activities).
    %   SYTest: Student test set (Scenes).
    %   TXTest: Teacher test set (Activities).
    %   TYTest: Teacher test set (Scenes).
    %   SceneTags: Scenes array.
    %   ClassNames: Classes array.

    disp("Loading data...")

    % Read Activities Seq File
    actSeq = readmatrix(actSeqFile,'OutputType','string','Delimiter',',');
    sceneSeq = readmatrix(sceneSeqFile,'OutputType','string','Delimiter',',');

    % Read Tags File
    actTags = readmatrix(actTagsFile,'OutputType','string','Delimiter',',');
    sceneTags = readmatrix(sceneTagsFile,'OutputType','string','Delimiter',',');

    % Read Set File
    sets = readmatrix(trainsetFile,'OutputType','string','Delimiter',',');

    % Init Train Set
    TXTrain = cell(0);
    SXTrain = cell(0);
    TYTrain = cell(0);
    SYTrain = cell(0);
    TXTest = cell(0);
    SXTest = cell(0);
    TYTest = cell(0);
    SYTest = cell(0);

    % Detect different classes
    classes = categories(categorical(actSeq(:,1)));
    classCount = size(classes,1);
    teacherTrainCount = 0;
    studentTrainCount = 0;
    classNames = [];

    for i = 1:classCount
        % Init class seq array
        className = classes{i,1};
        classNames = [classNames;className];
        teacherTestCount = 0;
        studentTestCount = 0;
        classSeqs = [];
        sceneSeqs = [];
        TXTest{i} = [];
        TYTest{i} = [];
        SXTest{i} = [];
        SYTest{i} = [];
        for j = 1:size(sceneSeq,1)
            if sceneSeq(j,1) == className
                sceneSeqs = categorical(sceneSeq(j,2:1:end),sceneTags);
                break;
            end
        end
        for j = 1:size(actSeq,1)
            if className == actSeq(j,1)
                colNum = size(actSeq(j,:),2);
                [r,c] = find(actSeq(j,:)=="");
                if size(c)>0
                    colNum = min(c) - 1;
                end
                classSeqs = [classSeqs;actSeq(j,2:1:colNum)];
            end
        end

        % Discover roles
        roleTypes = classSeqs(:,1);
        persons = classSeqs(:,2);
        actTypes = classSeqs(:,3);
        personNames = categories(categorical(persons));
        for k = 1:size(personNames,1)
            personName = personNames{k,1};
            roleType = "";
            actType = [];
            for l = 1:size(classSeqs,1)
                if classSeqs(l,2) == personName
                    roleType = classSeqs(l,1);
                    actType = [actType,classSeqs(l,3)];
                end
            end

            % Determine feature num
            features = [];
            for l = 1:size(actTags,1)
                if actTags(l,1) == roleType && ismember(actTags(l,2),actType)  
                    features = [features,actTags(l,3)];
                end
            end
            tmp = zeros(size(features,2),colNum - 4);
            for m = 1:size(actType,2)
                for frame = 1:size(tmp,2)
                    for n = 1:size(classSeqs,1)
                        if classSeqs(n,2) == personName
                            if(ismember(classSeqs(n,3+frame),features))
                                tmp(find(features==classSeqs(n,3+frame)),frame) = 1;
                            end
                        end
                    end
                end
            end

            % Seperate train / test set
            if roleType == "T"
                for m = 1:size(sets,1)
                    if className == sets(m,1) && personName == sets(m,2) && sets(m,3) == "Train"
                        teacherTrainCount = teacherTrainCount+1;
                        TXTrain{teacherTrainCount,1} = tmp;
                        TYTrain{teacherTrainCount,1} = sceneSeqs;
                        break;
                    elseif className == sets(m,1) && personName == sets(m,2) && sets(m,3) == "Test"
                        teacherTestCount = teacherTestCount+1;
                        TXTest{i,teacherTestCount,1} = tmp;
                        TYTest{i,teacherTestCount,1} = sceneSeqs;
                        break;
                    end
                end
            elseif roleType == "S"
                for m = 1:size(sets,1)
                    if className == sets(m,1) && personName == sets(m,2) && sets(m,3) == "Train"
                        studentTrainCount = studentTrainCount+1;
                        SXTrain{studentTrainCount,1} = tmp;
                        SYTrain{studentTrainCount,1} = sceneSeqs;
                        break;
                    elseif className == sets(m,1) && personName == sets(m,2) && sets(m,3) == "Test"
                        studentTestCount = studentTestCount+1;
                        SXTest{i,studentTestCount,1} = tmp;
                        SYTest{i,studentTestCount,1} = sceneSeqs;
                        break;
                    end
                end
            end
        end
    end  

    disp("Training...")

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
