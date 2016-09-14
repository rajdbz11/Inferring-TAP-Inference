function C = TAPCost(rMat,fMat,KVec,JMat)

[NVars,N_T,N_h] = size(rMat);

% Initialize C to zero
C = 0;

LUT = ...
    [0     0     0
     0     0     1
     0     0     2
     0     1     0
     0     1     1
     0     1     2
     0     2     0
     0     2     1
     0     2     2
     1     0     0
     1     0     1
     1     0     2
     1     1     0
     1     1     1
     1     1     2
     1     2     0
     1     2     1
     1     2     2
     2     0     0
     2     0     1
     2     0     2
     2     1     0
     2     1     1
     2     1     2
     2     2     0
     2     2     1
     2     2     2];
    

for hh = 1:N_h % For sessions
    
    GMat = zeros(NVars,N_T-1,27);
    
    for kk = 0:26 %     
        
        idx = LUT(kk+1,:);
        
        
        JMat_a = JMat.^idx(1);
        rMat_b = rMat(:,:,hh).^idx(2);
        rMat_c = rMat(:,:,hh).^idx(3);
        
%         JMat_a = JMat.^a;
%         rMat_b = rMat(:,:,hh).^b;
%         rMat_c = rMat(:,:,hh).^c;
        
        for tt = 1:N_T - 1
            % r = rMat(:,tt,hh);
            
            for V = 1:NVars
                % GMat(V,tt,kk+1) = KVec(kk+1)*(r(V).^b)*(JMat(:,V).^a)'*(r.^c);
                GMat(V,tt,kk+1) = KVec(kk+1)*rMat_b(V,tt)*JMat_a(:,V)'*rMat_c(:,tt);
            end
        end
    end
    
    gMat    = sum(GMat,3);
    fMatReq = fMat(:,:,hh);
    
    C = C + sum((gMat(:) - fMatReq(:)).^2);
    
    
end

