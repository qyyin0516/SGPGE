clear;
clc;

data = xlsread('orl_data.xlsx');
X = data(:, 1:1024);
y = data(:, 1025);

final_acc = [];
for iter = 1:10

    [x_train, y_train, x_test, y_test] = split_train_test(X, y, 40, 0.5);
    x_test = x_test';
    x_train = x_train';

    % sparsity represent
    [Feature_NUM, Train_NUM] = size(x_train);
    MatrixS = zeros(Train_NUM, Train_NUM);
    for i = 1:Train_NUM
        A = zeros(Feature_NUM, Train_NUM-1);
        if i == 1
            y = x_train(:, 1);
            A = x_train(:, 2:Train_NUM);
            x0 = inv(A' * A) * A' * y;
            xp = l1eq_pd(x0, A, [], y, 1e-3);
            xp = xp / norm(xp, 1);
            MatrixS(:, i) = [0 xp'];
        else
            y = x_train(:, i);
            A(:, 1:i-1) = x_train(:, 1:i-1);
            A(:, i:Train_NUM-1) = x_train(:, i+1:Train_NUM);
            x0 = inv(A' * A) * A' * y;
            xp = l1eq_pd(x0, A, [], y, 1e-3);
            xp = xp / norm(xp, 1);
            MatrixS(:, i) = [xp(1:i-1)' 0 xp(i:Train_NUM-1)'];
        end
    end

    % weight calculate
    W_pos = zeros(Train_NUM, Train_NUM);
    W_neg = zeros(Train_NUM, Train_NUM);
    W_loc = zeros(Train_NUM, Train_NUM);
    MatrixDist = zeros(Train_NUM, Train_NUM);
    tao_pos = zeros(1, Train_NUM);
    tao_neg = zeros(1, Train_NUM);
    tao_loc = zeros(1, Train_NUM);
    within_NUM = 10;
    between_NUM = Train_NUM - within_NUM;
    q = 1.5;
    k = 5;
    for i = 1:Train_NUM-1
        for j = i+1:Train_NUM
            dist_temp = [x_train(:, i)'; x_train(:, j)'];
            MatrixDist(i, j) = pdist(dist_temp) * pdist(dist_temp);
            MatrixDist(j, i) = MatrixDist(i, j);
        end
    end

    local_index = zeros(Train_NUM, k);
    local_dist = zeros(Train_NUM, k);
    for i = 1:Train_NUM
        [temp_dist, temp_index] = sort(MatrixDist(i, :));
        local_dist(i, :) = temp_dist(2:k+1);
        local_index(i, :) = temp_index(2:k+1);
    end
    sum_local = sum(local_dist, 2);
    for i = 1:Train_NUM
        tao_loc(i) = sum_local(i) / (k^2);
    end
    for i = 1:Train_NUM
        for j = 1:k
            if y_train(i) == y_train(local_index(i, j))
                W_loc(i, local_index(i, j)) = 0.5 * (exp(-local_dist(i, j) / tao_loc(i)) + exp(-local_dist(i, j) / tao_loc(j)));
            else
                W_loc(i, local_index(i, j)) = -0.5 * (exp(-local_dist(i, j) / tao_loc(i)) + exp(-local_dist(i, j) / tao_loc(j)));
            end
        end
    end
    for i = 1:Train_NUM
        sum_within = 0;
        sum_between = 0;
        within_NUM = 0;
        between_NUM = 0;
        for j = 1:Train_NUM
            if y_train(i) == y_train(j)
                sum_within = sum_within + MatrixDist(i, j);
                within_NUM = within_NUM + 1;
            else
                sum_between = sum_between + MatrixDist(i, j);
                between_NUM = between_NUM + 1;
            end
        end
        tao_pos(i) = sum_within / (within_NUM^q);
        tao_neg(i) = sum_between / (between_NUM^q);
    end
    for i = 1:Train_NUM
        for j = 1:Train_NUM
            if y_train(i) == y_train(j)
                W_pos(i, j) = 0.5 * (exp(-MatrixDist(i, j) / tao_pos(i)) + exp(-MatrixDist(i, j) / tao_pos(j)));
            else
                W_neg(i, j) = 0.5 * (exp(-MatrixDist(i, j) / tao_neg(i)) + exp(-MatrixDist(i, j) / tao_neg(j)));
            end
        end
    end

    % scatter_calculate
    H_pos = zeros(Train_NUM, Train_NUM);
    H_neg = zeros(Train_NUM, Train_NUM);
    H_loc = zeros(Train_NUM, Train_NUM);
    rowsum_pos = sum(W_pos, 2);
    rowsum_neg = sum(W_neg, 2);
    rowsum_loc = sum(W_loc, 2);
    for i = 1:Train_NUM
        H_pos(i, i) = rowsum_pos(i);
        H_neg(i, i) = rowsum_neg(i);
        H_loc(i, i) = rowsum_loc(i);
    end
    M_pos = H_pos - W_pos;
    M_neg = H_neg - W_neg;
    M_loc = H_loc - W_loc;   
    Sw = x_train * MatrixS * M_pos * MatrixS' * x_train';
    Sb = x_train * MatrixS * M_neg * MatrixS' * x_train';
    Sl = x_train * MatrixS * M_loc * MatrixS' * x_train';
    
    % GGPGE
    eig_mat = x_train * (M_neg - M_pos) * x_train';
    eig_mat = (eig_mat + eig_mat') / 2;
    [P, lambda] = eig(eig_mat);
    lambda_diag = diag(lambda);
    [lambda_sort, sort_index] = sort(lambda_diag, 'descend');
    lambda_sort = lambda_sort(sort_index);
    P_sort = P(:, sort_index);
    
    acc = [];
    acc_index = [];
    
    for i = 1:30
        P_reduce = P_sort(:, 1:i);
        train_new = x_train' * P_reduce;
        test_new = x_test' * P_reduce;
        knnModel = fitcknn(train_new, y_train, 'NumNeighbors', 1);
        y_pred = predict(knnModel, test_new);
        confusion = confusionmat(y_test, y_pred);
        acc = [acc, trace(confusion) / length(y_test)];
        acc_index = [acc_index, i];
    end

    final_acc = [final_acc; acc];
end
dimension_acc = mean(final_acc);
