% N = 20;
% 
% wVec = betarnd(2,2,[N,1]);
% WVec = wVec/sum(wVec);

WVec = [0.02357043  0.23218012  0.31823973  0.35058962  0.0754201]';
N = length(WVec);
U1 = rand(1,1)/N;
UVec = (U1:1/N:U1 + (N-1)/N)';

NVec = zeros(N,1);

for kk = 1:N
    if kk == 1
        LB = 0;
    else
        LB = sum(WVec(1:kk-1));
    end
    UB = sum(WVec(1:kk));
    idx1 = find(UVec >= LB);
    idx2 = find(UVec < UB);
    idx = intersect(idx1,idx2);
    
    NVec(kk) = length(idx);
end

