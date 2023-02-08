function [P_tX,P_Xt,pthat,features,log_P_Xt_shuffled,P_tX_chance] = ...
    naivebayestimedecoder(X,opt)
    %UNTITLED Summary of this function goes here
    %   Detailed explanation goes here

    %% input parsing
    [n_timepoints,n_features,~] = size(X);

    %% output preallocation
    features = cell(n_features,1);
    for ff = 1 : n_features
        features{ff} = struct();
    end
    features = cell2mat(features);
    X_mus = nan(n_timepoints,n_features);
    P_Xt = nan(n_timepoints,n_features,opt.n_xpoints);
    P_tX = nan(n_timepoints,n_timepoints,opt.test.n_trials);
    P_tX_chance = zeros(n_timepoints,n_timepoints,opt.test.n_trials);
    pthat.mode = nan(n_timepoints,opt.test.n_trials);
    pthat.median = nan(n_timepoints,opt.test.n_trials);
    pthat.mean = nan(n_timepoints,opt.test.n_trials);

    %% overwrite MATLAB's builtin definition for the poisson PDF
    poisspdf = @(lambda,k) ...
        exp(k .* log(lambda + 1e-100) - lambda - gammaln(k + 1));

    %% estimate feature supports

    % iterate through features
    for ff = 1 : n_features
        if opt.verbose
            progressreport(ff,n_features,'estimating feature supports');
        end

        % parse feature
        x = squeeze(X(:,ff,:));

        % estimate feature span
        x_bounds = quantile(x(:),[0,1]+[1,-1]*.001).*(1+[-1,1]*.025);
        [~,x_edges] = histcounts(x(x>=x_bounds(1)&x<=x_bounds(2)),opt.n_xpoints);

        % update feature
        features(ff).idx = ff;
        features(ff).x_bounds = x_bounds;
        features(ff).x_edges = x_edges;
        features(ff).x_bw = range(x_bounds) / 10;
    end
    
    %
    X_edges = vertcat(features.x_edges);

    %% construct encoding models

    % iterate through features
    for ff = 1 : n_features
        if opt.verbose
            progressreport(ff,n_features,'constructing encoding models');
        end

        % parse feature
        x = squeeze(X(:,ff,:));
        x_bounds = features(ff).x_bounds;
        x_edges = features(ff).x_edges;
        x_bw = features(ff).x_bw;

        % compute tuning function
        X_mus(:,ff) = nanmean(x(:,opt.train.trial_idcs),2);

        % kernel definition
        x_kernel = normpdf(x_edges,mean(x_bounds),x_bw);
        x_kernel = x_kernel / nansum(x_kernel);

        % compute joint distribution
        if opt.assumepoissonmdl

            % store theoretical joint distribution
            P_Xt(:,ff,:) = poisspdf(X_mus(:,ff),x_edges(1:end-1));
        else

            % preallocation
            p_Xc = nan(n_timepoints,opt.train.n_trials,opt.n_xpoints);

            % iterate through training trials
            for kk = 1 : opt.train.n_trials
                train_idx = opt.train.trial_idcs(kk);

                % compute likelihood
                x_counts = histcounts2(1:n_timepoints,x(:,train_idx)',...
                    'xbinedges',1:n_timepoints+1,...
                    'ybinedges',x_edges);
                p_Xc(:,kk,:) = conv2(1,x_kernel,x_counts,'same');
            end

            % store average empirical joint distribution
            P_Xt(:,ff,:) = nanmean(p_Xc,2);
        end

        % normalization
        P_Xt(:,ff,:) = P_Xt(:,ff,:) ./ nansum(P_Xt(:,ff,:),3);

        % update feature
        features(ff).x_mu = X_mus(:,ff);
        features(ff).p_Xc = squeeze(P_Xt(:,ff,:));
    end

    %%
    log_P_Xt = log(P_Xt);
    
    %% shuffling
%     % preallocation
%     shuffle_idcs = nan(n_timepoints,opt.test.n_trials,opt.shuffle.n);
%     
%     % sample shuffled time indices
%     for ss = 1 : opt.shuffle.n
%         if opt.verbose
%             progressreport(ss,opt.shuffle.n,'sampling shuffled indices');
%         end
%         shuffle_idcs(:,:,ss) = cell2mat(arrayfun(@(x)randperm(x),...
%             repmat(n_timepoints,opt.test.n_trials,1),...
%             'uniformoutput',false))';
%     end
    
    %% prior definition
    p_t = ones(n_timepoints,n_timepoints);
    
    % define gaussian kernel to introduce scalar timing
%     smearingkernel.win = opt.time;
%     smearingkernel.mus = smearingkernel.win';
%     smearingkernel.sigs = smearingkernel.mus * .25 + 0;
%     smearingkernel.pdfs = ...
%         normpdf(smearingkernel.win,smearingkernel.mus,smearingkernel.sigs);
%     I = eye(spkopt.time.roi.valid.len);
%     for ii = 1 : spkopt.time.roi.valid.len
%         if all(isnan(smearingkernel.pdfs(ii,:)))
%             smearingkernel.pdfs(ii,:) = I(ii,:);
%         end
%     end
%     smearingkernel.pdfs = smearingkernel.pdfs ./ nansum(smearingkernel.pdfs,2);

%     p_t = normpdf(opt.time,opt.time',opt.time'+.1);
%     p_t(:,opt.time < 0) = 1;
%     p_t(isnan(p_t)) = 0;
    
    % normalization
    p_t = p_t ./ nansum(p_t,1);
    
    %
    log_p_t = log(p_t);
    
    %% construct posteriors

    % iterate through test trials
    for kk = 1 : opt.test.n_trials
        if opt.verbose
            progressreport(kk,opt.test.n_trials,'constructing posteriors');
        end
        test_idx = opt.test.trial_idcs(kk);

        % iterate through time for the current test trial
        for tt = 1 : n_timepoints

            % fetch current observations
            x = X(tt,:,test_idx)';
            if all(isnan(x))
                continue;
            end
            
            % compute posterior for the current time point
            p_tX = decode(...
                x,X_edges,log_P_Xt,log_p_t(:,tt),n_features,n_timepoints);
            
            % store posterior
            P_tX(tt,:,kk) = p_tX;
        end
        
%         figure; hold on;
%         p_x = nan(n_timepoints,opt.shuffle.n);
% 
%         nan_flags = any(isnan(X(:,:,kk)),2);
%         shuffles = randsample(sum(~nan_flags),opt.shuffle.n,true);
%         
%         %
%         for ss = 1 : opt.shuffle.n
%             if opt.verbose
%                 progressreport(ss,opt.shuffle.n,'shuffling');
%             end
%             
%             % fetch shuffled observations
%             x = X(shuffles(ss),:,test_idx)';
%             if all(isnan(x))
%                 continue;
%             end
% 
%             p_x(:,ss) = predictnaivebayestimedecoder(...
%                 x,X_edges,P_Xt,p_t,n_features,n_timepoints);
%         end
%         
%         plot(opt.time,nanmean(p_x,2),'--k','linewidth',1.5);
%         plot(opt.time,nanmedian(p_x,2),'k','linewidth',1.5);
%         plot(opt.time,quantile(p_x,.25,2),'r','linewidth',1.5);
%         plot(opt.time,quantile(p_x,.95,2),'b','linewidth',1.5);
%         a=1;
%         continue;

%             % compute likelihoods of the current observations
%             if opt.assumepoissonmdl
% 
%                 % assume a features are poisson-distributed
%                 p_tx = poisspdf(X_mus',round(x));
%             else
% 
%                 % index current observation
%                 x_edges = vertcat(features.x_edges);
%                 [~,x_idcs] = min(abs(x_edges(:,1:end-1) - x),[],2);
% 
%                 % preallocation
%                 p_tx = nan(n_features,n_timepoints);
%                 p_x = nan(n_features,n_timepoints,opt.shuffle.n);
% 
%                 % iterate through features
%                 for ff = 1 : n_features
% 
%                     % assume empirical encoding model
%                     p_tx(ff,:) = P_Xt(:,ff,x_idcs(ff));
%                     
%                     % iterate through shuffles
%                     for ss = 1 : opt.shuffle.n
%                         
%                         % assume empirical encoding model
%                         p_x(ff,:,ss) = P_Xt(shuffle_idcs(:,test_idx,ss),ff,x_idcs(ff));
%                     end
%                 end
%             end
% 
%             figure;
%             hold on;
%             p_tX = prod(p_tx);
%             p_tX = p_tX ./ nansum(p_tX);
%             plot(p_tX);
% 
%             p_X = squeeze(prod(p_x));
%             p_X = p_X ./ nansum(p_X);
%             p_X_median = median(p_X,[1,2]);
%             p_X_mad = mad(p_X,0,[1,2]);
%             
%             plot(nanmean(p_X,2));
%             plot(xlim,[1,1]*mean(p_X,[1,2]));
%             plot(xlim,[1,1]*median(p_X,[1,2]),'k','linewidth',1);
%             plot(xlim,[1,1]*(median(p_X,[1,2]) + mad(p_X,0,[1,2])));
%             plot(xlim,[1,1]*(median(p_X,[1,2]) - mad(p_X,0,[1,2])));
%             a=1
%             
%             % normalization
%             p_tx = p_tx ./ nansum(p_tx,2);
%             nan_flags = all(isnan(p_tx),2) | isnan(x);
%             if all(nan_flags)
%                 continue;
%             end
% 
%             % compute posterior (accounting for numerical precision issues)
%             fudge = 1 + 1 / n_features;
%             p_tX = p_t .* prod(p_tx(~nan_flags,:) * n_timepoints)'; % * n_timepoints + fudge,1)';
% 
%             % normalization
%             P_tX(tt,:,kk) = p_tX / nansum(p_tX);
%             
%             %
%             p_X = squeeze(prod(p_x));
%             p_X = p_X ./ nansum(p_X);
%             p_X_median = median(p_X,[1,2]);
%             p_X_mad = mad(p_X,0,[1,2]);
% %             mask(tt,:,kk) = ...
% %                 P_tX(tt,:,kk) >= p_X_median + p_X_mad | ...
% %                 P_tX(tt,:,kk) <= p_X_median - p_X_mad;
%             mask(tt,:,kk) = p_X;
%         end

        % flag valid time for the current trial
        test_time_flags = ~all(isnan(P_tX(:,:,kk)),2);

        % fetch single trial posteriors to compute point estimates
        P_tX_kk = P_tX(test_time_flags,:,kk);

        % posterior mode (aka MAP)
        [~,mode_idcs] = max(P_tX_kk,[],2);
        pthat.mode(test_time_flags,kk) = opt.time(mode_idcs);

        % posterior median
        median_flags = [false(sum(test_time_flags),1),...
            diff(cumsum(P_tX_kk,2) > .5,1,2) == 1];
        [~,median_idcs] = max(median_flags,[],2);
        pthat.median(test_time_flags,kk) = opt.time(median_idcs);

        % posterior mean (aka COM)
        P_tX_kk(isnan(P_tX_kk)) = 0;
        pthat.mean(test_time_flags,kk) = opt.time * P_tX_kk';
    end
    
    %% shuffles
    
    % iterate through shuffles
    for ss = 1 : opt.n_shuffles
        
        % preallocation
        P_tX_shuffled = nan(n_timepoints,n_timepoints,opt.test.n_trials);

        % shuffle log-likelihoods along the time dimension
        log_P_Xt_shuffled = log_P_Xt(randperm(n_timepoints),:,:);
        
        %
        X_shuffled = X;
        for kk = opt.test.trial_idcs
            nan_flags = any(isnan(X_shuffled(:,:,kk)),2);
            X_shuffled(~nan_flags,:,kk) = ...
                X_shuffled(randperm(sum(~nan_flags)),:,kk);
        end
        
        % iterate through test trials
        for kk = 1 : opt.test.n_trials
            if opt.verbose
                progressreport(kk+(ss-1)*opt.test.n_trials,...
                    opt.test.n_trials*opt.n_shuffles,'shuffling');
            end
            test_idx = opt.test.trial_idcs(kk);
            
            % shuffle log-likelihoods along the time dimension
%             log_P_Xt_shuffled = log_P_Xt(randperm(n_timepoints),:,:);
            
            % iterate through time for the current test trial
            for tt = 1 : n_timepoints
                
                % fetch current observations
                x = X(tt,:,test_idx)';
                if all(isnan(x))
                    continue;
                end
                
                % compute posterior for the current time point
%                 p_tX = decode(...
%                     x,X_edges,log_P_Xt,log_p_t(:,tt),n_features,n_timepoints);
                p_tX = decode(...
                    x,X_edges,log_P_Xt_shuffled,log_p_t(:,tt),n_features,n_timepoints);
                
                % store posterior
                P_tX_shuffled(tt,:,kk) = p_tX;
            end
        end
        
        % add the posteriors of the current shuffle to the running average
        P_tX_chance = P_tX_chance + P_tX_shuffled / opt.n_shuffles;
    end
end

function mdl = train(X,opt)

end

function p_tX = decode(x,X_edges,log_P_Xt,log_p_t,n_features,n_timepoints)

    % index current observation
    [~,x_idcs] = min(abs(X_edges(:,1:end-1) - x),[],2);

    % preallocation
    log_p_tx = nan(n_features,n_timepoints);

    % iterate through features
    for ff = 1 : n_features

        % assume empirical encoding model
        log_p_tx(ff,:) = log_P_Xt(:,ff,x_idcs(ff));
    end
    
    % nan check
    nan_flags = all(isnan(log_p_tx),2) | isnan(x);
    if all(nan_flags)
        return;
    end

    % compute posterior by summing over log-likelihoods
    log_p_tX = log_p_t + nansum(log_p_tx(~nan_flags,:))';
    
    % convert back to probability
    p_tX = exp(log_p_tX - nanmax(log_p_tX));

    % normalization
    p_tX = p_tX / nansum(p_tX);
end