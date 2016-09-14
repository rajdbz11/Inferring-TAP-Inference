function [C] = CostFn(KVec,JMat,fMat,rMat)

[NVars,N_T,N_h] = size(rMat);

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
    
C = 0;               % Initialize Cost to zero
dKVec = zeros(27,1); % Initialize Grad to zero


for hh = 1:N_h % For sessions
    
    GMat = zeros(NVars,N_T-1,27);
    GamMat  = zeros(NVars,N_T-1,27);

    for kk = 0:26 %     
        
        a = LUT(kk+1,1);
        b = LUT(kk+1,2);
        c = LUT(kk+1,3);
         
        JMat_a = JMat.^a;
        rMat_b = rMat(:,:,hh).^b;
        rMat_c = rMat(:,:,hh).^c;
       
        
        for tt = 1:N_T - 1
            for V = 1:NVars
                GamMat(V,tt,kk+1)   = rMat_b(V,tt)*JMat_a(:,V)'*rMat_c(:,tt);
                GMat(V,tt,kk+1) = KVec(kk+1)*rMat_b(V,tt)*JMat_a(:,V)'*rMat_c(:,tt);
            end
        end
    end
    
    gMat    = sum(GMat,3); clear GMat;
    fMatReq = fMat(:,:,hh);
    
    C = C + sum((gMat(:) - fMatReq(:)).^2);
    
    for kk = 0:26      
        tempMat = (fMat(:,:,hh) - gMat).*GamMat(:,:,kk+1);
        dKVec(kk+1) = dKVec(kk+1) - 2*sum(sum(tempMat)); 
    end
      
end
