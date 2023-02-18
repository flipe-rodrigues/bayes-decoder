function [P_tX,P_Xt,pthat,features,log_P_Xt_shuff,P_tX_chance] = ...
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
    P_Xt_med = nan(n_timepoints,n_features,opt.n_xpoints);
    P_Xt_counts = nan(n_timepoints,n_features,opt.n_xpoints);
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
    
    % concatenate feature supports
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
            X_counts = nan(n_timepoints,opt.train.n_trials,opt.n_xpoints);
            
            % iterate through training trials
            for kk = 1 : opt.train.n_trials
                train_idx = opt.train.trial_idcs(kk);

                % compute likelihood
                x_counts = histcounts2(1:n_timepoints,x(:,train_idx)',...
                    'xbinedges',1:n_timepoints+1,...
                    'ybinedges',x_edges);
                
                % !!!!!!!
                nan_flags = isnan(x(:,train_idx));
                x_counts(nan_flags,:) = nan;

                %
                X_counts(:,kk,:) = x_counts;
            end

            % store average empirical joint distribution
            P_Xt_counts(:,ff,:) = nanmean(X_counts,2);
        end

        %
        bah = squeeze(P_Xt_counts(:,ff,:));
        bah = padarray(bah,[1,0]*opt.n_xpoints/2,'replicate','both');
        bah = padarray(bah,[0,1]*opt.n_xpoints/2,0,'both');
        post_avg_smoothing = conv2(x_kernel,x_kernel,bah,'valid');

%         figure('position',[1.8000 41.8000 1.0224e+03 472.8000]);
%         subplot(2,2,1); hold on;
%         a=no_norm_test; imagesc(opt.time,[],a'); axis tight;
%         subplot(2,2,2); hold on;
%         a=squeeze(P_Xt(:,ff,:)); imagesc(opt.time,[],a'); axis tight;
%         subplot(2,2,3); hold on;
%         b=post_avg_smoothing; imagesc(opt.time,[],b'); axis tight;
%         subplot(2,2,4); hold on;
%         c=squeeze(P_Xt_counts(:,ff,:)); imagesc(opt.time,[],c'); axis tight;
%         a=1
        
        P_Xt(:,ff,:) = post_avg_smoothing;
        
        % zero fix (to prevent -inf issues when "logging" afterwards)
        P_Xt_min = min(squeeze(P_Xt(:,ff,:)),[],1);
        epsilon = min(P_Xt_min(P_Xt_min>0),[],'all');
        P_Xt(:,ff,:) = P_Xt(:,ff,:) + epsilon;
%         
        % update feature
        features(ff).x_mu = X_mus(:,ff);
        features(ff).p_Xc = squeeze(P_Xt(:,ff,:));
    end

    %%
    log_P_Xt = log(P_Xt);
    
    %% prior definition

    %
    p_t = ones(n_timepoints,1) / n_timepoints;

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
%             p_tX = decode(...
%                 x,X_edges,log_P_Xt,log_p_t,n_features,n_timepoints);
            p_tX = decode2(...
                x,X_edges,P_Xt,log_P_Xt,log_p_t,n_features,n_timepoints);
            
            % store posterior
            P_tX(tt,:,kk) = p_tX;
        end
        
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
    if ~opt.shuffle
        return;
    end
    
    % iterate through shuffles
    for ss = 1 : opt.n_shuffles
        
        % preallocation
        P_tX_shuff = nan(n_timepoints,n_timepoints,opt.test.n_trials);

        % shuffle log-likelihoods along the time dimension
        log_P_Xt_shuff = log_P_Xt(randperm(n_timepoints),:,:);

        % iterate through test trials
        for kk = 1 : opt.test.n_trials
            if opt.verbose
                progressreport(kk+(ss-1)*opt.test.n_trials,...
                    opt.test.n_trials*opt.n_shuffles,'shuffling');
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
                    x,X_edges,log_P_Xt_shuff,log_p_t,n_features,n_timepoints);
                
                % store posterior
                P_tX_shuff(tt,:,kk) = p_tX;
            end
        end
        
        % add the posteriors of the current shuffle to the running average
        P_tX_chance = P_tX_chance + P_tX_shuff / opt.n_shuffles;
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
    
    %
    log_p_tx = log_p_tx - nanmax(log_p_tx,[],2);
    log_p_tx = log_p_tx ./ abs(nanmin(log_p_tx,[],2));
    
    % compute posterior by summing over log-likelihoods
    log_p_tX = log_p_t + nansum(log_p_tx(~nan_flags,:))';
    
    % exponentiate to get back to probability
    p_tX = exp(log_p_tX - nanmax(log_p_tX));

    % normalization
    p_tX = p_tX / nansum(p_tX);
end

function p_tX = decode2(x,X_edges,P_Xt,log_P_Xt,log_p_t,n_features,n_timepoints)

    % index current observation
    [~,x_idcs] = min(abs(X_edges(:,1:end-1) - x),[],2);

    % preallocation
    p_tx = nan(n_features,n_timepoints);
    log_p_tx = nan(n_features,n_timepoints);

    % iterate through features
    for ff = 1 : n_features

        % assume empirical encoding model
        p_tx(ff,:) = P_Xt(:,ff,x_idcs(ff));
        log_p_tx(ff,:) = log_P_Xt(:,ff,x_idcs(ff));
    end
    
    % nan check
    nan_flags = all(isnan(log_p_tx),2) | isnan(x);
    if all(nan_flags)
        return;
    end
    
    % inf check
    inf_flags = isinf(log_p_tx);
    log_p_tx(inf_flags) = nan;
    
%     % normalization
% %     log_p_tx = log_p_tx - nanmax(log_p_tx,[],2);
% %     log_p_tx = log_p_tx ./ abs(nanmin(log_p_tx,[],2));
%     p_tx = p_tx ./ nansum(p_tx,2);
%     log_p_tx2 = log(p_tx);
%     if ~all(log_p_tx == log_p_tx2);
%         a=1
%     end
    
    % compute posterior by summing over log-likelihoods
    log_p_tX = log_p_t + sum(log_p_tx(~nan_flags,:))';
    
    % exponentiate to get back to probability
    p_tX = exp(log_p_tX - nanmax(log_p_tX));

    % normalization
    p_tX = p_tX / nansum(p_tX);
    
    if any(isnan(p_tX))
        a=1
    end
end