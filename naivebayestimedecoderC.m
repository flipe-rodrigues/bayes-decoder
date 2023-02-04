classdef naivebayestimedecoderC
    %BAYESDECODER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        t;      % time
        p_t;    % prior    
        P_Xt;   % likelihoods
        P_tX;   % posteriors
    end
    
    methods
        function obj = naivebayestimedecoderC(t,p_t,P_Xt,P_tX)
            %BAYESDECODER Construct an instance of this class
            %   Detailed explanation goes here
            obj.t = t;
            obj.p_t = p_t;
            obj.P_Xt = P_Xt;
            obj.P_tX = P_tX;
        end
        
        function h = plotlikelihood(obj,idx)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            h = imagesc(obj.P_Xt(:,idx,:));
        end
    end
end

