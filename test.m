function [pred,scores,acc]= test(SNet,TNet,SXTest,SYTest,TXTest,TYTest,sceneTags,classNames)

    % Test once per class
    totalScore = [];
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
        acc = sum(pred == sytest)./numel(sytest);

        % Visualize
        figure
        plot(pred,'.-')
        hold on
        plot(sytest)
        title(classNames);
        hold off
    end
end