function [P_tX,P_Xt,pthat,features] = naivebayestimedecoder(tensor,opt)
    %UNTITLED Summary of this function goes here
    %   Detailed explanation goes here

    %% input parsing
    [n_timepoints,n_features,~] = size(tensor);

    %% output preallocation
    features = cell(n_features,1);
    for ff = 1 : n_features
        features{ff} = struct();
    end
    features = cell2mat(features);
    pthat.mode = nan(n_timepoints,opt.test.n_trials);
    pthat.median = nan(n_timepoints,opt.test.n_trials);
    pthat.mean = nan(n_timepoints,opt.test.n_trials);
    P_Xt = nan(n_timepoints,n_features,opt.n_xpoints);
    P_tX = nan(n_timepoints,n_timepoints,opt.test.n_trials);
    
    %% estimate feature spans

    % iterate through features
    for ff = 1 : n_features
        progressreport(ff,n_features,'estimating feature spans');

        % parse feature
        X = squeeze(tensor(:,ff,:));
        
        % estimate feature span
        x_bounds = quantile(X(:),[0,1]+[1,-1]*.01).*(1+[-1,1]*.05);
        [~,x_edges] = histcounts(X(X>=x_bounds(1)&X<=x_bounds(2)),opt.n_xpoints);
        
        % update feature
        features(ff).idx = ff;
        features(ff).x_bounds = x_bounds;
        features(ff).x_edges = x_edges;
        features(ff).x_bw = range(x_bounds) / 10;
    end
    
    %% construct encoding models

    % iterate through features
    for ff = 1 : n_features
        progressreport(ff,n_features,'constructing encoding models');
        
        % parse feature
        X = squeeze(tensor(:,ff,:));
        x_bounds = features(ff).x_bounds;
        x_edges = features(ff).x_edges;
        x_bw = features(ff).x_bw;

        % kernel definition
        x_kernel = normpdf(x_edges,mean(x_bounds),x_bw);
        x_kernel = x_kernel / nansum(x_kernel);
        
        % preallocation
        p_Xt = nan(n_timepoints,opt.train.n_trials,opt.n_xpoints);

        % iterate through training trials
        for kk = 1 : opt.train.n_trials
            train_idx = opt.train.trial_idcs(kk);
            
            % compute likelihood
            x_counts = histcounts2(1:n_timepoints,X(:,train_idx)',...
                'xbinedges',1:n_timepoints+1,...
                'ybinedges',x_edges);
            p_Xt(:,kk,:) = conv2(1,x_kernel,x_counts,'same');
        end
        
        % store average joint distribution
        P_Xt(:,ff,:) = nanmean(p_Xt,2);
    end

    % normalization
    P_Xt = P_Xt ./ nansum(P_Xt,3);

    %% construct posteriors

    % prior definition
    p_t = ones(n_timepoints,1) / n_timepoints;
    
    % iterate through test trials
    for kk = 1 : opt.test.n_trials
        progressreport(kk,opt.test.n_trials,'constructing posteriors');
        test_idx = opt.test.trial_idcs(kk);

        % iterate through true time for the current test trial
        for tt = 1 : n_timepoints
            
            % fetch current observations
            x = tensor(tt,:,test_idx)';
            if all(isnan(x))
                continue;
            end

            % index current observation
            x_edges = vertcat(features.x_edges);
            [~,x_idcs] = min(abs(x_edges(:,1:end-1) - x),[],2);
            
            % preallocation
            p_tx = nan(n_features,n_timepoints);
            
            % iterate through features
            for ff = 1 : n_features
                
                % assume empirical encoding model
                p_tx(ff,:) = P_Xt(:,ff,x_idcs(ff));
            end
            
            % normalization
            p_tx = p_tx ./ nansum(p_tx,2);
            nan_flags = all(isnan(p_tx),2);
            if all(nan_flags)
                continue;
            end
            
            % compute posterior (with numerical precision issues in mind)
            fudge = 1 + 1 / n_features;
            p_tX = p_t .* prod(p_tx(~nan_flags,:) * n_timepoints + fudge,1)';
            p_X = nansum(p_tX);
            P_tX(tt,:,kk) = p_tX / p_X;
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
end