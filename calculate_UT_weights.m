function W = calculate_UT_weights(n)

W = zeros(2*n+1,1);
W(1) = 0.1;
W(2:end) = (1-W(1))/2/n;
end