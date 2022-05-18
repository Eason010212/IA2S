function [analyzeRes] = analyze(X,Y,Net,minProb,minLength,sceneTags)

    % Function: Analyze activity seqs, carry out seqs that describe certain
    %           person's status (1=normal, 0=abnormal) based on scene
    %           probability.

    % Usage: [analyzeRes] = analyze(X,Y,Net,minProb,minLength,sceneTags)

    % Author: Song Yishen @ CIT Lab

    % Input:
    %   X: Certain people's activity seqs.
    %   Y: Actual scene seqs.
    %   Net: Predict model related to peoples' role type.
    %   minProb: Minimum confidence probability. When the probability that 
    %            the prediction result is an actual scene category is less 
    %            than this value, it is judged to be abnormal activity.
    %   minLength: The shortest abnormal length. When the abnormal 
    %              activity continuously appears in more than the number of
    %              frames exceeding this value, the abnormal activity is 
    %              considered as output, otherwise it is still judged as 
    %              normal activity.
    %   SceneTags: Scene tags array.

    % Output:
    %   analyzeRes: Analyze result cell array.

    % Init cell array
    analyzeRes = cell(0);

    conLength = 0;
    for i = 1:size(X,1)
        for j = 1:size(X,2)
            tmp = [];
            [pred, scores] = classify(Net,X{i,j,1});
            for k = 1:size(scores,2)
                % Assume current frame is normal
                tmp(1,k) = 1;
                % Detect abnormal probability
                prob = scores(find(categorical(sceneTags) == pred(1,k)),k);
                if prob < minProb
                    conLength = conLength+1;
                    % Determine abnormal sequence
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