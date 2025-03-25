function [F] = updateF(A, X, L, W, F, K, G, lambda1)

cls_num = size(F, 2);
V = length(X);
n = size(X{1}, 2);

% formulation of B
B = zeros(n, n);
for v = 1:V
    B = B + K{v}^2 + lambda1*G{v}'*L{v}*G{v};
end
B = 2 .* B;

% formulation of Y
Y = zeros(n,cls_num);
for v = 1:V
    Y = Y - (K{v}^2)*X{v}'*W{v}*A;
end

% solving F
opts = [];  opts.info = 0;
opts.gtol = 1e-5;
[F, ~] = FOForth(F, Y, @fun, opts, B, Y);
    function [funX, T] = fun(X, B, Y)
        % f(X): = 0.5*trace(X'AX) + trace(G'X)
        T = B * X + Y;
        funX = 0.5 * sum(sum(X .* (T + Y)));
    end
end