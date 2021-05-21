%% Make data for Deep NN Eq

% Montana State University
% Electrical & Computer Engineering Department
% Created by Alexander Salois

for i = 1:30
  [x,y,x_val,y_val] = make_data_deep(i,5e3);
  savename = sprintf('deepSNR%02d',i)
  save(savename,'x','y','x_val','y_val');
end
