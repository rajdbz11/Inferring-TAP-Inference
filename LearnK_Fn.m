function KVec = LearnK_Fn(fMat, JMat, rMat)

% For now, lot of hard-coding. a, b, c take values {0,1,2}. 
% Therefore, there are 3x3x3 = 27 K_abc coefficients. We therefore have a
% system of 27 equations to solve for. 

[NVars, N_T, N_H] = size(rMat);

KMat = zeros(27,N_H);

for hh = 1:N_H
% First compute the Gamma matrices for each abc 

GCell = {};

for kk = 0:26 % 27-1
    abc = dec2base(kk,3,3);
    a   = str2double(abc(1));
    b   = str2double(abc(2));
    c   = str2double(abc(3));
    
    GMat = zeros(NVars,N_T-1);
    for tt = 1:N_T-1
        r = rMat(:,tt,hh);
        for V = 1:NVars
            GMat(V,tt) = (r(V).^b)*(JMat(:,V).^a)'*(r.^c);
        end
    end
    GCell{kk+1} = GMat;   
end

% Now set up the system of equations 

BVec = zeros(27,1);

for kk = 1:27
    GMat = GCell{kk};
    BVec(kk) = sum(sum(fMat(:,:,hh).*GMat));
end

AMat = zeros(27,27);

for ii = 1:27
    GMat_i = GCell{ii};
    for jj = 1:ii
        GMat_j = GCell{jj};
        AMat(ii,jj) = sum(sum(GMat_i.*GMat_j));
        AMat(jj,ii) = sum(sum(GMat_i.*GMat_j));
    end
end

KVec = AMat\BVec;
KMat(:,hh) = KVec;

end

KVec = mean(KMat,2);
