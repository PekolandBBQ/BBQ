09%% 建立一个DNN分类器
%% 赋值数据集
%onehot数据
Xclsallonehot=[
    onehotWorkorderType,...% 1/业务类型：
    onehotWorkordersonType,...% 2/业务子类
    onehotAcceptancetime,...% 3/受理时间 小时数
    onehotDaytype,...% 4/受理日类型
    onehotUrgency,...% 5/受理紧急程度
    onehotImportance,...% 6/受理重要程度
    onehotFirstappraise,...% 7/首次评议
    onehotTalktime,...% 8/通话时长
    onehotSupplyarea,...% 9/供电所类型
    onehotElectricitype,...% 10/用电类型
    onehotTriptime,...% 11-18/前一天、两天、三天重合闸成功、不成功
    onehotTransformer,...% 19/受理变电站编号
    onehotHalfyearworkorder,...% 20/受理用户前半年总工单
    onehotHalfyearnorm2complaint,...% 21/受理用户前半年一般工单提级投诉
    onehotHalfyearopinion2opinion,...% 22/受理用户前半年意见不提级工单
    onehotHalfyearopinion2complaint,...% 23/受理用户前半年意见转投诉工单
    onehotHalfyearcomplaint,...% 24/受理用户前半年投诉工单
    onehotTemperatureday,...% 25-27/受理当天温度：最大/最小/平均
    onehotTemperaturedaybefore,...% 28-30/受理前期三天温度：最大/最小/平均
    onehotTemperaturedayafter,...% 31-33/受理后期三天温度：最大/最小/平均
    onehotWeathertypeday,...% 34-36/受理当天/前三天/后三天最恶劣天气
    onehotWindpowerday,...% 37-39/受理当天最大/最小/平均风力
    onehotWindpowerdaybefore,...% 40-42受理前三天最大/最小/平均风力
    onehotWindpowerdayafter,...% 43-45受理后三天平均最大/最小/风力
    onehotWinddirection,...% 46/受理当天风向
    onehotPowerload,...% 47-49/受理前三天最大/最小/平均负荷
];


%对onehot数据降维
% [Xclsallcoeff,Xclsallscore,Xclsalllatent,Xclsalltsquared,Xclsallexplained]= pca(Xclsallonehot,'Centered' ,'off');
% idx = find(cumsum(Xclsallexplained)>98,1);
% Xclsallonehotdedim=Xclsallscore(:,1:idx);

%标签数据
Xclsalllabel=[
    labelWorkorderType,...% 1/业务类型：
    labelWorkordersonType,...% 2/业务子类
    labelAcceptancetime,...% 3/受理时间 小时数
    labelDaytype,...% 4/受理日类型
    labelUrgency,...% 5/受理紧急程度
    labelImportance,...% 6/受理重要程度
    labelFirstappraise,...% 7/首次评议
    labelTalktime,...% 8/通话时长
    labelSupplyarea,...% 9/供电所类型
    labelElectricitype,...% 10/用电类型
    labelTriptime,...% 11-18/前一天、两天、三天重合闸成功、不成功
    labelTransformer,...% 19/受理变电站编号
    labelHalfyearworkorder,...% 20/受理用户前半年总工单
    labelHalfyearnorm2complaint,...% 21/受理用户前半年一般工单提级投诉
    labelHalfyearopinion2opinion,...% 22/受理用户前半年意见不提级工单
    labelHalfyearopinion2complaint,...% 23/受理用户前半年意见转投诉工单
    labelHalfyearcomplaint,...% 24/受理用户前半年投诉工单
    labelTemperatureday,...% 25-27/受理当天温度：最大/最小/平均
    labelTemperaturedaybefore,...% 28-30/受理前期三天温度：最大/最小/平均
    labelTemperaturedayafter,...% 31-33/受理后期三天温度：最大/最小/平均
    labelWeathertypeday,...% 34-36/受理当天/前三天/后三天最恶劣天气
    labelWindpowerday,...% 37-39/受理当天最大/最小/平均风力
    labelWindpowerdaybefore,...% 40-42受理前三天最大/最小/平均风力
    labelWindpowerdayafter,...% 43-45受理后三天平均最大/最小/风力
    labelWinddirection,...% 46/受理当天风向
    labelPowerload,...% 47-49/受理前三天最大/最小/平均负荷
];


%value数据
Xclsallvalue=[
    valueWorkorderType,...% 1/业务类型：
    valueWorkordersonType,...% 2/业务子类
    valueAcceptancetime,...% 3/受理时间 小时数
    valueDaytype,...% 4/受理日类型
    valueUrgency,...% 5/受理紧急程度
    valueImportance,...% 6/受理重要程度
    valueFirstappraise,...% 7/首次评议
    valueTalktime,...% 8/通话时长
    valueSupplyarea,...% 9/供电所类型
    valueElectricitype,...% 10/用电类型
    valueTriptime,...% 11-18/前一天、两天、三天重合闸成功、不成功
    valueTransformer,...% 19/受理变电站编号
    valueHalfyearworkorder,...% 20/受理用户前半年总工单
    valueHalfyearnorm2complaint,...% 21/受理用户前半年一般工单提级投诉
    valueHalfyearopinion2opinion,...% 22/受理用户前半年意见不提级工单
    valueHalfyearopinion2complaint,...% 23/受理用户前半年意见转投诉工单
    valueHalfyearcomplaint,...% 24/受理用户前半年投诉工单
    valueTemperatureday,...% 25-27/受理当天温度：最大/最小/平均
    valueTemperaturedaybefore,...% 28-30/受理前期三天温度：最大/最小/平均
    valueTemperaturedayafter,...% 31-33/受理后期三天温度：最大/最小/平均
    valueWeathertypeday,...% 34-36/受理当天/前三天/后三天最恶劣天气
    valueWindpowerday,...% 37-39/受理当天最大/最小/平均风力
    valueWindpowerdaybefore,...% 40-42受理前三天最大/最小/平均风力
    valueWindpowerdayafter,...% 43-45受理后三天平均最大/最小/风力
    valueWinddirection,...% 46/受理当天风向
    valuePowerload,...% 47-49/受理前三天最大/最小/平均负荷
];

%过采样
%onehot数据70k*5
% XclsallonehotOversampling=[XclsallonehotcategoryoneOversampling(100000:700000,:);
%     XclsallonehotcategorytwoOversampling(1:700000,:);
%     XclsallonehotcategorythreeOversampling(1:700000,:),...
%     XclsallonehotcategoryfourOversampling(1:700000,:),...
%     XclsallonehotcategoryfiveOversampling(1:700000,:)];

%label数据70k*5
XclsalllabelOversampling=[XclsalllabelcategoryoneOversampling(200000:700000,:);
     XclsalllabelcategorytwoOversampling(1:700000,:);
     XclsalllabelcategorythreeOversampling(1:700000,:);
     XclsalllabelcategoryfourOversampling(1:700000,:);
     XclsalllabelcategoryfiveOversampling(1:700000,:)];


%value数据70k*5
XclsallvalueOversampling=[XclsallvaluecategoryoneOversampling(200000:700000,:);
     XclsallvaluecategorytwoOversampling(1:700000,:);
     XclsallvaluecategorythreeOversampling(1:700000,:);
     XclsallvaluecategoryfourOversampling(1:700000,:);
     XclsallvaluecategoryfiveOversampling(1:700000,:)];

%归一化输入
 [Xclsalllabelstdtrans,xps]=[mapminmax(XclsallvalueOversampling',0,1)];
Xclsalllabelstd=Xclsalllabelstdtrans';
%classification层输出的初始样本，待转化为


Yclsallonehot=[onehotWorkordercategory
    ];%输出层是softmax

Yclsallvalue=[valueWorkordercategory
    ];%输出层是分类层
Yclsalllabel=[labelWorkordercategory   
    ];%输出层是分类层

%过采样
%onehot数据70k*5
% YclsallonehotOversampling=[YclsallonehotcategoryoneOversampling(100000:700000,:);
%     YclsallonehotcategorytwoOversampling(1:700000,:);
%     YclsallonehotcategorythreeOversampling(1:700000,:);
%     YclsallonehotcategoryfourOversampling(1:700000,:);
%     YclsallonehotcategoryfiveOversampling(1:700000,:)];

%label数据70k*5
YclsalllabelOversampling=[YclsalllabelcategoryoneOversampling(200000:700000,:);
     YclsalllabelcategorytwoOversampling(1:700000,:);
     YclsalllabelcategorythreeOversampling(1:700000,:);
     YclsalllabelcategoryfourOversampling(1:700000,:);
     YclsalllabelcategoryfiveOversampling(1:700000,:)];


%value数据70k*5
YclsallvalueOversampling=[YclsallvaluecategoryoneOversampling(200000:700000,:);
     YclsallvaluecategorytwoOversampling(1:700000,:),;
     YclsallvaluecategorythreeOversampling(1:700000,:);
     YclsallvaluecategoryfourOversampling(1:700000,:);
     YclsallvaluecategoryfiveOversampling(1:700000,:)];

%处理后的数据
Xclsallstd=Xclsalllabelstd;%Xclsallonehotdedim
Yclsallstd=categorical(YclsallvalueOversampling);%输出层是分类层
%% 分离训练样本、测试样本 
% holdout分割
% [TrainIndexprimary,TestIndexprimary] = crossvalind('HoldOut',size(Xclsallstd,1),0.15);
% 
% XclsTrainprimary=Xclsallstd(TrainIndexprimary,:); %random第一次分割
% YclsTrainprimary=Yclsallstd(TrainIndexprimary,:); %random第一次分割
% [TrainIndexsecond,TestIndexsecond]=crossvalind('HoldOut',size(XclsTrainprimary,1),0.5);
% %50% 欠采样
% XclsTrain=Xclsallstd(TrainIndexsecond,:); %random第二次分割
% YclsTrain=Yclsallstd(TrainIndexsecond,:); %random第二次分割
% 
% %测试样本：2021年4月1日至2021年4月31日
% XclsTest=Xclsallstd(TestIndexprimary,:);  %random第一次分割
% YclsTest=Yclsallstd(TestIndexprimary,:);  %random第一次分割

featureindex=[1:3,5:49];
%顺序分割
TrainIndex=[1:1+450000,...
    500000:500000+650000,...
    500000+700000:500000+700000+650000,...
    500000+700000*2:500000+700000*2+650000,...
    500000+700000*3:500000+700000*3+650000];
TestIndex=[450000:500000,...
    500000+650000:500000+700000,...
    500000+700000+650000:500000+700000*2,...
    500000+700000*2+650000:500000+700000*3,...
    500000+700000*3+650000:500000+700000*4];

 XclsTrain=Xclsallstd(TrainIndex,featureindex);%inorder
 YclsTrain=Yclsallstd(TrainIndex,:); %inorder

 XclsTest=Xclsallstd(TestIndex,featureindex); %inorder
 YclsTest=Yclsallstd(TestIndex,:); %inorder

%% 选择训练参数
clsoptions = trainingOptions('sgdm', ...%训练算法设定
    'ExecutionEnvironment','parallel', ...%训练硬件设定       
     'LearnRateSchedule','piecewise', ...%学习率参数
     'InitialLearnRate',0.002, ...
    'LearnRateDropPeriod',3, ...
    'LearnRateDropFactor',0.95, ...
    'L2Regularization',0.001,...
    'MaxEpochs',2000, ...%最大训练次数
    'MiniBatchSize',30000, ...%小批量
    'SequenceLength','longest', ...
    'Shuffle','every-epoch', ...
    'Verbose',0, ...%绘图与显示    
    'Plots','training-progress');%泛化目标精度
  
%      , ...
%     'ValidationData',{XclsTest,YclsTest}, ...%泛化集
%       'ValidationFrequency',10, ...%泛化频率
%       'ValidationPatience',inf

%% 定义 LSTM 网络架构
inputSize = size(XclsTrain,2);
outputSize = 5;
numHiddenUnits1 = 150;
numHiddenUnits2 = 70;
numHiddenUnits3 = 30;

numfast=100;

%clslayers是基础性网络
 clslayerGraph = [ ...
    featureInputLayer(inputSize,'name','feainput')...  
    fullyConnectedLayer(numHiddenUnits1,'name','full1')...
    reluLayer('name','relu1')...
    fullyConnectedLayer(numHiddenUnits2,'name','full2')...%
    reluLayer('name','relu2')....%additionLayer(2,'name','add')...   
    fullyConnectedLayer(numHiddenUnits3,'name','full3')...%
    reluLayer('name','relu3')....
    dropoutLayer(0.05,'name','drop1')...
    fullyConnectedLayer(outputSize,'name','fulloutput')...
    softmaxLayer('name','sfm')...
    classificationLayer('name','cls')];
%   
%    clslayerGraph=layerGraph(clslayers);
% %   fastfull=[fullyConnectedLayer(numHiddenUnits3,'name','fastfull')...
% %   clippedReluLayer(1,'name','clpfast')];  
% % clslayerGraph = addLayers(clslayerGraph,fastfull);  
% %   clslayerGraph=connectLayers(clslayerGraph,'seq','fastfull');
% %  clslayerGraph=connectLayers(clslayerGraph,'clpfast','add/in2');
%  clslayerfigure= figure('name','clslayerGraph')  ;
%   plot(clslayerGraph)
%% 训练 DNN 分类预测模型

[clsfeaturenet,tr] = trainNetwork(XclsTrain,YclsTrain,clslayerGraph,clsoptions);

%% lime测试
blackbox=@(x)classify(clsfeaturenet,x);

NumImportantPredictors=size(XclsTrain,2);

results = lime(blackbox,XclsTrain,'NumImportantPredictors',NumImportantPredictors,'Type','classification','KernelWidth',0.1,'SimpleModelType','tree');

contImportantPredictor=zeros(NumImportantPredictors,1);

 for i=1:size(XclsTrain,1)
 singleresult=fit(results,XclsTrain(i,:),NumImportantPredictors,'KernelWidth',0.1,'SimpleModelType','tree');
 singleImportantPredictorsindex=singleresult.ImportantPredictors;
 for j=1:size(singleImportantPredictorsindex)
     contindex=singleImportantPredictorsindex(j);
    contImportantPredictor(contindex)=contImportantPredictor(contindex)+1;
 end
 end
 
  %% 测试网络/分类
 
  YclsPred=classify(clsfeaturenet,XclsTest);

 %计算分类准确性
  %acc = sum(YclsPred==YclsTest)./numel(YclsTest);

 %计算分类准确性
  %acc = sum(YclsPredtreated==YclsTesttreated)./numel(YclsTest);
%   YclsPred=categorical(YclsPred);%转换为类别矩阵
%   YclsTest=categorical(YclsTest);%转换为类别矩阵
  
 %创建混淆矩阵
 confusionfigure=figure('name','confusion');
 YclsTestwithname=categorical(double(YclsTest),[1 ,2 ,3 , 4, 5],{'一般工单无提级风险','一般工单提级','意见工单无提级风险','意见工单提级','投诉工单'});
 YclsPredwithname=categorical(double(YclsPred),[1 ,2 ,3, 4, 5],{'一般工单无提级风险','一般工单提级','意见工单无提级风险','意见工单提级','投诉工单'});
 plotconfusion(YclsTestwithname,YclsPredwithname) ;
  %绘图
   numsample=numel(double(YclsTest));
  scatterfigure= figure('name','scatter')  ;
        
          scatter(1:numsample,double(YclsPred),15,'r','.')
          hold on  
          
          scatter(1:numsample,double(YclsTest),15,'k','square')
          hold on 
          
  axis([ 0 numel(YclsPred) 0 6])
  set(scatterfigure.CurrentAxes,'yticklabel',{'','一般工单无提级风险','一般工单提级','意见工单无提级风险','意见工单提级','投诉工单',''},'ytick',[0:6]);
  xlabel('95598工单序号（按时间排序）')
  ylabel('95598工单类别')
  legend('预测类别','实际类别') 
%  grid on
  title('基于DBN的95598工单风险类别预测')
 