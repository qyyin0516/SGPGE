clc;
clear;

for FF = 1:30
    for KK = 1:10
        I = xlsread('orl_data.xlsx');
        X = I(:, 1:1024);
        y = I(:, 1025);
        Image_row_NUM = 32;
        Image_column_NUM = 32; 
        NN = Image_row_NUM * Image_column_NUM;

        Class_Train_NUM = 5;
        Class_Sample_NUM = 10; % total
        Class_Test_NUM = Class_Sample_NUM - Class_Train_NUM;

        Class_NUM = 40;
        Train_NUM = Class_NUM * Class_Train_NUM; % 
        Test_NUM = Class_NUM * (Class_Sample_NUM - Class_Train_NUM); % 

        [Train_DAT, y_train, Test_DAT, y_test] = split_train_test(X, y, 40, 0.5);

        Train_DAT = Train_DAT';
        Test_DAT = Test_DAT';

        % to center each training sample and testing sample
        Mean_Image = mean(Train_DAT, 2);  
        Train_DAT = Train_DAT - Mean_Image * ones(1, Train_NUM);
        Test_DAT = Test_DAT - Mean_Image * ones(1, Test_NUM);

        Train_DAT = Train_DAT';
        Test_DAT = Test_DAT';

        [projMatrix, D, Gmd] = myLDA(Train_DAT, y_train, 40, FF);

        % LDA Transform:
        Train_SET = Train_DAT * projMatrix; % size of (Eigen_NUM, Train_NUM); % PCA-based 
        Test_SET = Test_DAT * projMatrix;   % size of (Eigen_NUM, Test_NUM); % PCA-based

        %% do classification using 1NN
        model = fitcknn(Train_SET, y_train, 'NumNeighbors', 1);
        predict_label = predict(model, Test_SET);
        accuracy(KK) = length(find(predict_label == y_test)) / length(y_test);
    end
    mean_accur(FF) = mean(accuracy);
end

function [projMatrix, D, Gmd] = myLDA(data, label, class, d)
    if nargin < 4
        d = 1;
    end

    cn = length(class);
    dm = size(data, 2);

    if d >= dm
        error('Projection dimension must be smaller than original data dimension...\n');
    end

    % compute total mean values of data
    u = mean(data, 1);
    % compute mean value of each class and within-class variation
    Sw = zeros(dm, dm);
    Sb = zeros(dm, dm);
    for ci = 1:cn
        id = find(label == class(ci));
        ni = length(id);

        % mean value and variation of class ci
        ui = mean(data(id, :), 1);
        var = data(id, :) - repmat(ui, ni, 1);
        Sw = Sw + var' * var;                  % within-class variation    
        Sb = Sb + ni * (ui - u)' * (ui - u);   % between-class variation
    end

    % train the projection matrix
    [W, D] = eig(Sb, Sw);
    [~, I] = sort(diag(D), 'descend');
    projMatrix = W(:, I(1:d));

    % compute data matrix after projection
    projData = data * projMatrix;
    % compute mean and cov of gaussian distribution of each class
    Gmd.U = zeros(d, cn);
    Gmd.C = zeros(d, d, cn);
    for ci = 1:cn
        id = find(label == class(ci));
        Gmd.U(:, ci) = mean(projData(id, :), 1);
        Gmd.C(:, ci) = cov(projData(id, :));
    end
end
