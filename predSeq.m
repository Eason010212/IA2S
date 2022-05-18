function [predResults] = predSeq(SNet,TNet,SXPredict,TXPredict,sceneTags,classNames)

    % Function: Use trained network and activity seqs to predict scene
    %           seqs.

    % Usage: [predResults] = pred(SNet,TNet,SXPredict,TXPredict,sceneTags,
    %                        classNames)
    
    % Author: Song Yishen @ CIT Lab

    % Input:
    %   SNet: Trained LSTM Net using student sequences.
    %   TNet: Trained LSTM Net using teacher sequences.
    %   SXPredict: Student predict set (Activities).
    %   TXPredict: Teacher predict set (Activities).
    %   SceneTags: Scene tags array.
    %   ClassNames: Classes array.

    % Output:
    %   predResults: Predicted scenes cell array.

    % Init results
    predResults = cell(0);

    for i = 1:size(classNames,1)
        totalScore = [];

        % Use student net to predict
        for j = 1:size(SXPredict,2)
            sxpredict = SXPredict{i,j,1};
            [pred, scores] = classify(SNet,sxpredict);
            if size(totalScore)==0
                totalScore = scores;
            else
                totalScore = totalScore+scores;
            end
        end

        % Use teacher net to predict
        for j = 1:size(TXPredict,2)
            txpredict = TXPredict{i,j,1};
            [pred, scores] = classify(TNet,txpredict);
            if size(totalScore)==0
                totalScore = scores;
            else
                totalScore = totalScore+scores;
            end
        end

        % Joint prediction
        pred = [];
        for j = 1:size(totalScore,2)
            col = totalScore(:,j);
            pred = [pred,sceneTags(find(col==max(col)),1)];
        end
        pred = categorical(pred,sceneTags);
        predResults{i,1} = pred;
    end
end