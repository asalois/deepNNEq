%% Make data for Deep NN Eq

% Montana State University
% Electrical & Computer Engineering Department
% Created by Alexander Salois
tic
for i = 1:40
    [x,y,x_val,y_val] = make_data_deep(i,5e3);
    savename = sprintf('deepSNR%02d',i)
    [x_test, seq] = predict_data(i,2^22);
    save(savename,'x','y','x_val','y_val','x_test','seq');
    toc
end
