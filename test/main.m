[SXTrain,SYTrain,TXTrain,TYTrain,SXTest,SYTest,TXTest,TYTest,SXPredict,TXPredict,sceneTags,classNames] = readData("activities.csv","scenes.csv","actTags.csv","sceneTags.csv","set.csv");
[SNet,TNet] = train(SXTrain,SYTrain,TXTrain,TYTrain,sceneTags);
[preds,acc] = test(SNet,TNet,SXTest,SYTest,TXTest,TYTest,sceneTags,classNames);
[predResults] = predSeq(SNet,TNet,SXPredict,TXPredict,sceneTags,classNames);
[analyzeRes] = analyze(SXTest,SYTest,SNet,0.50,10,sceneTags);