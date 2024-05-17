clc;
clear;

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

    Train_DAT = zeros(Train_NUM, NN);
    Test_DAT = zeros(Test_NUM, NN);

    [Train_DAT, y_train, Test_DAT, y_test] = split_train_test(X, y, 40, 0.5);

    Train_DAT = Train_DAT';
    Test_DAT = Test_DAT';

    % to center the each training sample and testing sample
    Mean_Image = mean(Train_DAT, 2);  
    Train_DAT = Train_DAT - Mean_Image * ones(1, Train_NUM);
    Test_DAT = Test_DAT - Mean_Image * ones(1, Test_NUM);

    %% do dimensional reduction, by PCA pre-processing

    % Eigen_NUM can be changed! Eigen_NUM must be less than Train_NUM
    Eigen_NUM = 1;
    [PCA_Projection, disc_value] = Eigenface_f(Train_DAT, Eigen_NUM);   

    % LLE_DP Transform:
    Train_SET = PCA_Projection' * Train_DAT; % size of (Eigen_NUM, Train_NUM); % PCA-based 
    Test_SET = PCA_Projection' * Test_DAT;   % size of (Eigen_NUM, Test_NUM); % PCA-based

    %% Feature extraction using SPP
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Step 1, Construct weight matrix S using MSR, Eq(15). That is SPP1
    MatrixS = zeros(Train_NUM, Train_NUM);
    for k = 1:1:Train_NUM
        A = zeros(Eigen_NUM, Train_NUM - 1);
        if k == 1
            y = Train_SET(:, 1);
            A = Train_SET(:, 2:Train_NUM);
            x0 = inv(A' * A) * A' * y;
            % solve the L1_minimization through y=Ax
            xp = l1eq_pd(x0, A, [], y, 1e-3);
            xp = xp / norm(xp, 1);
            MatrixS(:, k) = [0, xp'];
        else
            y = Train_SET(:, k);
            A(:, 1:k-1) = Train_SET(:, 1:k-1);
            A(:, k:Train_NUM-1) = Train_SET(:, k+1:Train_NUM);
            x0 = inv(A' * A) * A' * y;
            xp = l1eq_pd(x0, A, [], y, 1e-3);
            xp = xp / norm(xp, 1);
            MatrixS(:, k) = [xp(1:k-1)', 0, xp(k:Train_NUM-1)'];
        end
        clear A;
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Step 2, using Eq.(22) to calculate the projection W
    E = eye(Train_NUM, Train_NUM);
    SB = MatrixS + MatrixS' - MatrixS' * MatrixS;

    Mat1 = Train_SET * SB * Train_SET';
    Mat2 = Train_SET * Train_SET';

    % d is the number of projections, and d must be no larger than Eigen_NUM
    d = 27;

    [W, Gen_Value] = Find_K_Max_Gen_Eigen(Mat1, Mat2, Eigen_NUM); % solve equations to get weight vectors
    Train = W' * Train_SET;
    Train = Train';
    Test = W' * Test_SET;
    Test = Test';

    %% do classification using 1NN
    model = ClassificationKNN.fit(Train, y_train, 'NumNeighbors', 1);
    predict_label = predict(model, Test);
    accuracy(KK) = length(find(predict_label == y_test)) / length(y_test);
end

SPP_1NN_acc = mean(accuracy);


function [Eigen_Vector, Eigen_Value] = Find_K_Max_Gen_Eigen(Matrix1, Matrix2, Eigen_NUM)

    [NN, NN] = size(Matrix1);

    % Note this is equivalent to; [V, S] = eig(St, SL); also equivalent to [V, S] = eig(Sn, St); %
    [V, S] = eig(Matrix1, Matrix2); % eig(A,B) returns the N generalized eigenvalues of square matrices A and B in a

    S = diag(S); % Extract diagonal elements into a column vector
    [S, index] = sort(S); % Sort the elements of S in ascending order

    Eigen_Vector = zeros(NN, Eigen_NUM);
    Eigen_Value = zeros(1, Eigen_NUM);

    p = NN; % Select the largest Eigen_NUM eigenvalues and their corresponding eigenvectors
    for t = 1:Eigen_NUM
        Eigen_Vector(:, t) = V(:, index(p));
        Eigen_Value(t) = S(p);
        p = p - 1;
    end
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


