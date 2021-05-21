function [x,y,x_val,y_val] = make_data_deep(SNR,chunks)
%deepnnEq Make and train an Deep NN EQ
%   Detailed explanation goes here

samples = 4;
num_pow = 10;
val_size = floor(log2( (chunks * (2^num_pow)) /8 ));

[data, target] = get_train_data(num_pow,SNR,samples);
[x_val, y_val] = get_train_data(val_size,SNR,samples);

x = zeros(size(data,1),size(data,2)*chunks);
y = zeros(size(target,1),size(target,2)*chunks);

for iter = 1:chunks
    x(:,((iter-1)*size(data,2)+1):(iter*size(data,2))) = data;
    y(:,((iter-1)*size(target,2)+1):(iter*size(target,2))) = target;
    if iter ~= chunks
        [data, target] = get_train_data(num_pow,SNR,samples);
    end
end
x = x';
y = y';
x_val = x_val';
y_val = y_val';


end
