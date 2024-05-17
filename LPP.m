clear;
clc;

data = xlsread('orl_data.xlsx');
X = data(:, 1:1024);
y = data(:, 1025);

for d = 1:30
    [eigvector, ~] = myLPP(X, d);
    M = X * eigvector;

    kf = 10;
    indices = crossvalind('Kfold', 400, kf);
    for p = 1:kf
        test = (indices == p);
        train = ~test;
        train_data = M(train, :);
        train_label = y(train);
        test_data = M(test, :);
        test_label = y(test);

        mdl = ClassificationKNN.fit(train_data, train_label, 'NumNeighbors', 1);
        predict_label = predict(mdl, test_data);
        accuracy(p) = length(find(predict_label == test_label)) / length(test_label);
    end
    accur(d) = mean(accuracy);
end

function [eigvector, eigvalue] = myLPP(X, d)
    % X is the matrix that every row is a sample and every column is a feature 
    X = X';
    N = size(X, 2); % numbers of sample vectors
    W = single(zeros(N, N));
    D = single(zeros(N, N));
    L = single(zeros(N, N));
    for i = 1:N
        for j = 1:N
            a = X(:, i);
            b = X(:, j);
            W(i, j) = dot(a, b) / (norm(a) * norm(b));
        end
    end
    for i = 1:N
        temp = 0;
        for j = 1:N
            temp = temp + W(i, j);
        end
        D(i, i) = temp;
    end
    L = D - W;
    A = X * L * X';
    B = X * D * X';
    [V, lamda] = eig(A, B);
    n = size(lamda, 1);
    lamda = lamda * ones(n, 1); % change lamda to a line vector
    valueSet = zeros(n, 1);
    num = 1;
    for i = 1:n
        valueSet(i, 1) = num;
        num = num + 1;
    end
    map = containers.Map(lamda, valueSet);
    map.sort();
    temp = values(map);
    v = [];
    for i = 1:size(temp, 2)
        v = [v V(:, temp{i})];
    end
    v = v(:, 50:50 + d);
    eigvector = v;
    eigvalue = lamda;
end
