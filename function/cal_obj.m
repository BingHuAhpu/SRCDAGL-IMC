function [ loss_sum, beta ] = cal_obj( X, H, A, w, M)
V = length(X);
loss = zeros(V, 1);
for v = 1:V
    loss(v) = sum(sum((X{v} - A{v}*H').^2)) + sum(sum(w{v}*(H*H' - M{v}*M{v}')).^2);
end
loss_sum = w'*loss;

end

