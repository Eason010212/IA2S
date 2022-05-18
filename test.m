function [preds,acc] = test(SNet,TNet,SXTest,SYTest,TXTest,TYTest,sceneTags,classNames)
    
    % Function: Use trained network and test set to test accuracy.

    % Usage: [predResults,acc] = test(SNet,TNet,SXTest,SYTest,TXTest,TYTest,
    %                     sceneTags,classNames)
    
    % Author: Song Yishen @ CIT Lab

    % Input:
    %   SNet: Trained LSTM Net using student sequences.
    %   TNet: Trained LSTM Net using teacher sequences.
    %   SXTest: Student test set (Activities).
    %   SYTest: Student test set (Scenes).
    %   TXTest: Teacher test set (Activities).
    %   TYTest: Teacher test set (Scenes).
    %   SceneTags: Scene tags array.
    %   ClassNames: Classes array.

    % Output:
    %   predResults: Predicted results on test sets.
    %   acc: Accuracy array on test sets.
    
    % Test once per class
    preds = cell(0);

    totalScore = [];
    acc = [];
    for i = 1:size(classNames,1)
        className = classNames(i,:);

        % Probability from students seqs
        for j = 1:size(SXTest,2)
            sxtest = SXTest{i,j,1};
            if size(sxtest)~=0
                sytest = SYTest{i,j,1};
                [pred, scores] = classify(SNet,sxtest);
                if size(totalScore)==0
                    totalScore = scores;
                else
                    totalScore = totalScore+scores;
                end
            end
        end

        % Probability from teacher seqs
        for j = 1:size(TXTest,2)
            txtest = TXTest{i,j,1};
            if size(txtest)~=0
                tytest = TYTest{i,j,1};
                [pred, scores] = classify(TNet,txtest);
                if size(totalScore)==0
                    totalScore = scores;
                else
                    totalScore = totalScore+scores;
                end
            end
        end

        % Joint Prediction
        pred = [];
        for j = 1:size(totalScore,2)
            col = totalScore(:,j);
            pred = [pred,sceneTags(find(col==max(col)),1)];
        end
        pred = categorical(pred,sceneTags);
        preds{i,1} = pred;
        acc = [acc;sum(pred == sytest)./numel(sytest)];

        % Visualize
        figure
        plot(pred,'.-')
        hold on
        plot(sytest)
        title(classNames);
        hold off
    end
end