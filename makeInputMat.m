function mat = makeInputMat(input,numSamples)
%makeInputMat Makes in input matrix
%   Detailed explanation goes here
l = 2*numSamples + 1;
mat = zeros(l,length(input));
for i = 1:l
    mat(i,:) = circshift(input,-i);
end
mat = round([real(mat); imag(mat)],5);
end

