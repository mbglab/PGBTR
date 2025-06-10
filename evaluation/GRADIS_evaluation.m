% Before using this code you should install GRADIS from https://github.com/MonaRazaghi/GRADIS
% You can change "Alldata" and "path_prefix" to the dataset which you want to evaluate.

clear, clc

%% Loading data

disp('Loading data...     (takes a while)');

%% Alldata is the GDmatrix.csv which will be acquired after you run the Preprocessing_evaluation.py .
Alldata = readtable('D:/subject/pycharm/CNNITR/data/Bsubtilis/GDmatrix_bin32.csv','ReadVariableNames', true);

disp('Loading data ...     (Done!)');

%% Final svm
final_auroc = [];
final_aupr = [];
final_precision = [];
final_recall = [];
final_f1score = [];
for iter_num =6:10
    path_prefix = ['D:/subject/pycharm/CNNITR/data/Bsubtilis/processed_balanced/',num2str(iter_num)];
    auroc_list = [];
    aupr_list = [];
    precision_list = [];
    recall_list = [];
    f1score_list = [];
    for n =0:4
        test_file = [path_prefix,'/test_data', num2str(n), '.txt'];
        train_file = [path_prefix,'/train_data', num2str(n), '.txt'];
        Test_group = readtable(test_file, 'Delimiter', '\t');
        Train_group = readtable(train_file, 'Delimiter', '\t');

        indexColumn1 = Test_group{:, 1};
        indexColumn2 = Test_group{:, 2};
        indexColumn3 = Train_group{:, 1};
        indexColumn4 = Train_group{:, 2};

        logicalIndices1 = [];
        logicalIndices2 = [];
        
        for i = 1:size(Test_group, 1)
            rowIndices = find(strcmp(table2array(Alldata(:, 1)), indexColumn1{i}) & strcmp(table2array(Alldata(:, 2)), indexColumn2{i}));
            if ~isempty(rowIndices)
                logicalIndices1 = [logicalIndices1,rowIndices];
            end
        end
        
        for i = 1:size(Train_group, 1)
            rowIndices = find(strcmp(table2array(Alldata(:, 1)), indexColumn3{i}) & strcmp(table2array(Alldata(:, 2)), indexColumn4{i}));
            if ~isempty(rowIndices)
                logicalIndices2 = [logicalIndices2,rowIndices];                               
            end
        end

        Test_data = Alldata(logicalIndices1, :);
        Train_data = Alldata(logicalIndices2, :);

        SVMModel = fitcsvm(Train_data(:,3:end), Train_group(:,3),'Holdout',0.2,'Standardize',true,'KernelFunction','rbf',...
            'KernelScale','auto');
        CompactSVMModel = SVMModel.Trained{1}; % Extract trained, compact classifier


        % Test
        [label,score] = predict(CompactSVMModel,Test_data(:,3:end));
        accuracy = sum(label == table2array(Test_group(:,3)))/height(Test_data)*100;

        [X_AUC,Y_AUC,Tsvm,AUCsvm] = perfcurve(logical(table2array(Test_group(:,3))),score(:,logical(CompactSVMModel.ClassNames)),'true');
        [X_PR,Y_PR,~,PRsvm] = perfcurve(logical(table2array(Test_group(:,3))),score(:,logical(CompactSVMModel.ClassNames)),'true','xCrit', 'reca', 'yCrit', 'prec');

        %% Display results

        C = confusionmat(table2array(Test_group(:,3)), label);
        disp(C)
        TP = C(2,2);  
        FN = C(2,1);  
        Recall = TP / (TP + FN);
        TP = C(2,2);  
        FP = C(1,2);  
        Precision = TP / (TP + FP);
        F1_Score = 2 * (Precision * Recall) / (Precision + Recall);

        auroc_list = [auroc_list, AUCsvm];    
        aupr_list = [aupr_list, PRsvm];
        precision_list = [precision_list, Precision];
        recall_list = [recall_list, Recall];
        f1score_list = [f1score_list, F1_Score];
    end
    STR1 = [num2str(auroc_list), '\nThe area under ROC is : ', num2str(mean(auroc_list))];
    disp(STR1);
    STR2 = [num2str(aupr_list), '\nThe area under PR curve is : ', num2str(mean(aupr_list))];
    disp(STR2);
    
    final_auroc = [final_auroc, mean(auroc_list)];
    final_aupr = [final_aupr, mean(aupr_list)];
    final_precision = [final_precision, mean(precision_list)];
    final_recall = [final_recall, mean(recall_list)];
    final_f1score = [final_f1score, mean(f1score_list)];
    
    
end
STR3 = [num2str(final_auroc), ' ROC : ', num2str(mean(final_auroc)),'��', num2str(std(final_auroc))];
disp(STR3);
STR4 = [num2str(final_aupr), ' PR : ', num2str(mean(final_aupr)),'��', num2str(std(final_aupr))];
disp(STR4);
STR5 = [num2str(final_precision), ' Precision : ', num2str(mean(final_precision))];
disp(STR5);
STR6 = [num2str(final_recall), ' Recall : ', num2str(mean(final_recall))];
disp(STR6);
STR7 = [num2str(final_f1score), ' f1-score : ', num2str(mean(final_f1score)), '��', num2str(std(final_f1score))];
disp(STR7);
