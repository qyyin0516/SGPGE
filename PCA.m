clc;
clear;

PCA_1NN_acc = zeros(30, 1); 
for FF = 1:30
    accuracy = zeros(10, 1);
    
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

        % to center the each training sample and testing sample
        Mean_Image = mean(Train_DAT, 2);
        Train_DAT = Train_DAT - Mean_Image * ones(1, Train_NUM);
        Test_DAT = Test_DAT - Mean_Image * ones(1, Test_NUM);
        
        % do dimensional reduction, by PCA
        % Eigen_NUM can be changed! Eigen_NUM must be less than Train_NUM
        Eigen_NUM = FF;
        [PCA_Projection, ~] = Eigenface_f(Train_DAT, Eigen_NUM);

        % LLE_DP Transform:
        Train_SET = PCA_Projection' * Train_DAT; % size of (Eigen_NUM,Train_NUM); % PCA-based
        Test_SET = PCA_Projection' * Test_DAT; % size of (Eigen_NUM,Test_NUM); % PCA-based

        Train_SET = Train_SET';
        Test_SET = Test_SET';

        %% do classification using 1NN
        model = fitcknn(Train_SET, y_train, 'NumNeighbors', 1);
        predict_label = predict(model, Test_SET);
        accuracy(KK) = sum(predict_label == y_test) / length(y_test);
    end
    
    PCA_1NN_acc(FF) = mean(accuracy);
end


function [disc_set, disc_value, Mean_Image] = Eigenface_f(Train_SET, Eigen_NUM)

    % the magnitude of eigenvalues of this function is corrected right 
    % Centralized PCA
    [NN, Train_NUM] = size(Train_SET);

    if NN <= Train_NUM % for small sample size case
        
        Mean_Image = mean(Train_SET, 2);  
        Train_SET = Train_SET - Mean_Image * ones(1, Train_NUM);
        R = Train_SET * Train_SET' / (Train_NUM - 1);
       
        [V, S] = Find_K_Max_Eigen(R, Eigen_NUM);
        disc_value = S;
        disc_set = V;
       
    % for small sample size case
    else 
        
        Mean_Image = mean(Train_SET, 2); 
        Train_SET = Train_SET - Mean_Image * ones(1, Train_NUM);
        
        R = Train_SET' * Train_SET / (Train_NUM - 1); 
        [V, S] = Find_K_Max_Eigen(R, Eigen_NUM); 
        
        disc_value = S;
        disc_set = zeros(NN, Eigen_NUM);
        Train_SET = Train_SET / sqrt(Train_NUM - 1);
        
        for k = 1:Eigen_NUM
            disc_set(:, k) = (1 / sqrt(disc_value(k))) * Train_SET * V(:, k); 
        end
    end
end

function [Eigen_Vector, Eigen_Value] = Find_K_Max_Eigen(Matrix, Eigen_NUM)

    [NN, ~] = size(Matrix);

    % Note this is equivalent to; [V, S] = eig(St, SL); also equivalent to [V, S] = eig(Sn, St); %
    [V, S] = eig(Matrix); 

    S = diag(S); 
    [S, index] = sort(S);

    Eigen_Vector = zeros(NN, Eigen_NUM);
    Eigen_Value = zeros(1, Eigen_NUM);

    p = NN;
    for t = 1:Eigen_NUM
        Eigen_Vector(:, t) = V(:, index(p));
        Eigen_Value(t) = S(p);
        p = p - 1;
    end
end

