function [analyzeRes] = analyze(X,Y,Net,minProb,minLength,sceneTags)
    analyzeRes = cell(0);
    conLength = 0;
    for i = 1:size(X,1)
        for j = 1:size(X,2)
            tmp = [];
            [pred, scores] = classify(Net,X{i,j,1});
            for k = 1:size(scores,2)
                % Assume current frame is normal
                tmp(1,k) = 1;
                
                prob = scores(find(categorical(sceneTags) == pred(1,k)),k);
                if prob < minProb
                    conLength = conLength+1;
                    if(conLength>=minLength)
                        tmp(1,k) = 0;
                    end
                else
                    conLength = 0;
                end
            end
            analyzeRes{i,j,1} = tmp;
        end
    end
end