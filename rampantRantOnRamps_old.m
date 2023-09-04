%% initialization
warning('off');
close all;
clear;
clc;

%% notes
% beta depends on time units (s or ms), and firing rate;
% rho and beta are redundant;

%% seed fixing
% rng(0);

%% tensor settings
T = 100;    % time points
N = 15;     % neurons
K = 1e3;    % trials
B = 1;      % bootstrap iterations

%% time settings
ti = 1;
tf = T;
t = linspace(ti,tf,T);
dt = diff(t(1:2));

%% "ramping" criteria
rho_cutoff = .5;
beta_cutoff = .05;
pval_cutoff = .05;

%% variability parameters
gammas = exprnd(1,N,1);
lambdas = exprnd(.05*(tf-ti),N,1);
mus = linspace(ti,tf,N)';
sigmas = repmat(.25*(tf-ti),N,1);
% phis = abs(normrnd(0,1,N,1));

%% firing rate function
x_fun = @(gamma,mu,lambda,sigma) gamma * normpdf(t,normrnd(mu,lambda),sigma);

%% color settings
clrs = cool(N);

%% bootstrapping

% preallocation
SD_ramp = nan(B,1);
SD_non = nan(B,1);

% iterate through bootstrap iterations
for bb = 1 : B
    progressreport(bb,B,'bootstrapping')
    
    %% generate fake data
    
    % rate settings
    gains = rand(1,N) .^ 1;
    mus = linspace(ti,tf,N)';
    sigmas = .25 * tf;
    
    % noise settings
    noise = normrnd(0,.25,T,N) * .01 * 0;
    
    % generate fake rates
    x = normpdf(t,mus,sigmas)' .* gains + noise;
    
    % normalize to max
    max_fr = 15;
    x = (x - min(x,[],'all')) / max(x,[],'all') * max_fr;
    
    %% sample spike trains
    
    % preallocation
    X = nan(T,N,K);
    spiketimes = cell(N,K);
    
    % iterate through neurons
    for nn = 1 : N
        progressreport(nn,N,'generating spike data');
        for kk = 1 : K
            X(:,nn,kk) = x_fun(gammas(nn),mus(nn),lambdas(nn),sigmas(nn));
            dur = tf - ti - kernel.paddx(1);
            [n,ts] = poissonprocess(X(:,nn,kk),dur);
            spiketimes{nn,kk} = ts + ti + kernel.paddx(1);
        end
    end
    
    return;
    
    %% normalization
    z = zscore(x);
    
    %% compute correlation coefficients
    rhos = nan(N,1);
    
    % iterate through neurons
    for nn = 1 : N
        rhos(nn) = corr(t',x(:,nn));
    end
    
    %% perform linear regression
    mdls = cell(N,1);
    pvals = nan(N,1);
    betas = nan(N,1);
    
    % iterate through neurons
    for nn = 1 : N
        mdls{nn} = fitlm(t,x(:,nn));
        p = polyfit(t,x(:,nn),1);
        pvals(nn) = mdls{nn}.Coefficients.pValue(end);
        betas(nn) = p(1); % mdls{nn}.Coefficients.Estimate(end);
    end
    
    %% flag "ramping" neurons
    ramp_flags = ...
        abs(rhos) > rho_cutoff & ...
        abs(betas) > beta_cutoff & ...
        pvals <= pval_cutoff;
    
    %% naive bayes decoder
    x_ramp = cat(3,x(:,ramp_flags),x(:,ramp_flags));
    x_non = cat(3,x(:,~ramp_flags),x(:,~ramp_flags));
    % x_non(:,:,2) = interp1(t,x_non(:,:,2),t*1.5);
    
    opt = struct();
    opt.n_xpoints = 100;
    opt.time = t;
    opt.train.trial_idcs = 1;
    opt.train.n_trials = numel(opt.train.trial_idcs);
    opt.test.trial_idcs = 2;
    opt.test.n_trials = numel(opt.test.trial_idcs);
    opt.assumepoissonmdl = true;
    opt.verbose = false;
    
    % preallocation
    P_Rt = nan(T,N,opt.n_xpoints);
    [P_tR_ramp,P_Rt(:,ramp_flags,:),~,neurons(ramp_flags)] = ...
        naivebayestimedecoder(x_ramp,opt);
    [P_tR_non,P_Rt(:,~ramp_flags,:),~,neurons(~ramp_flags)] = ...
        naivebayestimedecoder(x_non,opt);
    P_tR_ramp = P_tR_ramp ./ nansum(P_tR_ramp,1);
    P_tR_non = P_tR_non ./ nansum(P_tR_non,1);
    
    %% plot encoding models
    % figure;
    % set(gca,...
    %     'xlim',[ti,tf],...
    %     'xdir','normal',...
    %     'ydir','normal',...
    %     'nextplot','add',...
    %     'plotboxaspectratio',[1,1,1]);
    % xlabel('Time (a.u.)');
    % ylabel('Rate (a.u.)');
    %
    % % iterate through units
    % for nn = 1 : N
    %     cla;
    %     r_bounds = neurons(nn).x_bounds;
    %     r_bw = neurons(nn).x_bw;
    %     if range(r_bounds) == 0
    %         continue;
    %     end
    %     ylim(r_bounds);
    %     title(sprintf('Neuron: %i, bw: %.2f',nn,r_bw));
    %     p_Rt = squeeze(P_Rt(:,nn,:));
    %     p_Rt(isnan(p_Rt)) = max(p_Rt(:));
    %     imagesc([ti,tf],r_bounds,p_Rt');
    %     plot(t,neurons(nn).x_mu,...
    %         'color','w',...
    %         'linewidth',1);
    %     drawnow;
    %     pause(.1);
    % end
    % close;
    
    %% compute decoding error
    
    % preallocation
    mu_ramp = nan(T,1);
    mu_non = nan(T,1);
    sd_ramp = nan(T,1);
    sd_non = nan(T,1);
    
    % iterate through time points
    for tt = 1 : T
        mu_ramp(tt) = P_tR_ramp(:,tt)' * t';
        mu_non(tt) = P_tR_non(:,tt)' * t';
        sd_ramp(tt) = sqrt(P_tR_ramp(:,tt)' * (mu_ramp(tt) - t') .^ 2);
        sd_non(tt) = sqrt(P_tR_non(:,tt)' * (mu_non(tt) - t') .^ 2);
    end
    
    % store current bootstrap
    SD_ramp(bb) = nanmean(sd_ramp);
    SD_non(bb) = nanmean(sd_non);
    
    %%
    if bb < B
        continue;
    end
    
    %% tiling
    
    % figure initialization
    fig = figure(...
        'color','w',...
        'name','tiling');
    
    % axes initialization
    set(gca,...
        'xlim',[ti,tf],...
        'ylim',[1,N],...
        'ytick',[1,N],...
        'colormap',hot(2^8),...
        'layer','top',...
        'tickdir','out',...
        'nextplot','add',...
        'plotboxaspectratio',[1,1,1],...
        'linewidth',2,...
        'fontsize',12,...
        'ticklength',[1,1]*.025);
    xlabel('Time (s)');
    ylabel('Neuron #');
    
    % color limits
    clim = [-2,4];
    
    % plot psth raster
    % imagesc(t,[1,N],z',clim);
    imagesc(t,[1,N],x');
    % imagesc(t,[1,N],x_non(:,:,2)');
    
    %% plot single-neuron averages
    figure(...
        'position',[300,350,750,420],...
        'color','w');
    set(gca,...
        'xlim',[ti,tf],...
        'ylim',[0,max_fr] + [-1,1] * .05 * max_fr,...
        'ycolor','none',...
        'layer','top',...
        'tickdir','out',...
        'nextplot','add',...
        'plotboxaspectratio',[1,1,1],...
        'linewidth',2,...
        'fontsize',12,...
        'ticklength',[1,1]*.025);
    xlabel('time (a.u.)');
    
    % iterate through neurons
    for nn = 1 : N
        
        % plot single-neuron average
        plot(t,x(:,nn),...
            'color',clrs(nn,:),...
            'linewidth',1);
        
        % plot model prediction
        plot(t,mdls{nn}.predict(t'),...
            'color',clrs(nn,:),...
            'linestyle',repmat('-',1+(pvals(nn)>pval_cutoff),1),...
            'linewidth',1);
        
        % annotate correlation coefficient
        text(1.05,nn/N,sprintf('\\rho = %.2f',rhos(nn)),...
            'color',clrs(nn,:),...
            'horizontalalignment','left',...
            'units','normalized');
        
        % annotate regression statistics
        text(-.05,nn/N,...
            sprintf('\\beta = %.2f, p-value = %.2f',betas(nn),pvals(nn)),...
            'color',clrs(nn,:),...
            'horizontalalignment','right',...
            'units','normalized');
        
        % afford ramping class
        if ramp_flags(nn)
            plot(mus(nn),max(ylim),...
                'marker','v',...
                'markerfacecolor',clrs(nn,:),...
                'markeredgecolor','w',....
                'markersize',7.5,...
                'linewidth',1);
        end
    end
    
    %% plot posterior averages
    figure(...
        'name','condition-split posterior averages',...
        'numbertitle','off',...
        'windowstyle','docked');
    n_rows = 1;
    n_cols = 2;
    sps = gobjects(n_rows,n_cols);
    for rr = 1 : n_rows
        for cc = 1 : n_cols
            sp_idx = cc + (rr - 1) * n_cols;
            sps(rr,cc) = subplot(n_rows,n_cols,sp_idx);
            xlabel(sps(rr,cc),'real time (a.u.)');
            ylabel(sps(rr,cc),'decoded time (a.u.)');
        end
    end
    set(sps,...
        'xlim',[ti,tf],...
        'ylim',[ti,tf],...
        'xdir','normal',...
        'ydir','normal',...
        'nextplot','add',...
        'plotboxaspectratio',[1,1,1]);
    linkaxes(sps);
    
    % iterate through conditions
    clims = quantile([P_tR_ramp(:);P_tR_non(:)],[0,1]);
    
    % ramping posteriors
    title(sps(1),sprintf('Ramping neurons (%i/%i)',sum(ramp_flags),N));
    imagesc(sps(1),[ti,tf],[ti,tf],P_tR_ramp',clims);
    plot(sps(1),xlim(sps(1)),ylim(sps(1)),'-k');
    plot(sps(1),xlim(sps(1)),ylim(sps(1)),'--w');
    
    % non-ramping posteriors
    title(sps(2),sprintf('Non-ramping neurons (%i/%i)',sum(~ramp_flags),N));
    imagesc(sps(2),[ti,tf],[ti,tf],P_tR_non',clims);
    plot(sps(2),xlim(sps(2)),ylim(sps(2)),'-k');
    plot(sps(2),xlim(sps(2)),ylim(sps(2)),'--w');
    
    %%
    
    figure;
    hold on;
    xlim([ti,tf]);
    
    for tt = 1 : T
        cla;
        plot(t,P_tR_ramp(:,tt),...
            'color','k',...
            'linewidth',1.5);
        plot(mu_ramp(tt),max(P_tR_ramp(:,tt)),...
            'color','k',...
            'marker','v',...
            'markersize',7.5,...
            'linewidth',1.5);
        plot(mu_ramp(tt)+[-1,1]*sd_ramp(tt),[1,1]*max(P_tR_ramp(:,tt)),...
            'color','k',...
            'linewidth',1.5);
        
        plot(t,P_tR_non(:,tt),...
            'color','r',...
            'linewidth',1.5);
        plot(mu_non(tt),max(P_tR_non(:,tt)),...
            'color','r',...
            'marker','v',...
            'markersize',7.5,...
            'linewidth',1.5);
        plot(mu_non(tt)+[-1,1]*sd_non(tt),[1,1]*max(P_tR_non(:,tt)),...
            'color','r',...
            'linewidth',1.5);
        
        if B == 1
            pause(.05)
        end
    end
    %%
    % figure;
    % hold on;
    %
    % x_test = linspace(0,1,T);
    % x_pdf = normpdf(x_test,.35,.05);
    % x_pdf = x_pdf / nansum(x_pdf);
    %
    % x_mu = x_pdf * x_test';
    % x_sd = sqrt(x_pdf * (x_mu - x_test') .^ 2);
    %
    % plot(x_test,x_pdf);
    % plot(x_mu,max(x_pdf),...
    %     'color','r',...
    %     'marker','v',...
    %     'markersize',7.5,...
    %     'linewidth',1.5)
    % plot(x_mu+[-1,1]*x_sd,[1,1]*max(x_pdf),...
    %     'color','r',...
    %     'linewidth',1.5)
    
    %%
    
    figure;
    hold on;
    xlabel('SD P(t | R_{ramping})');
    ylabel('SD P(t | R_{non-ramping})');
    
    plot(sd_ramp,sd_non,'.');
    plot(sd_ramp(1),sd_non(1),'ok');
    plot(nanmean(sd_ramp),nanmean(sd_non),'xr');
    
    % [~,idx] = max(P_tR_non,[],2);
    % plot(t,t(idx),'linewidth',2);
    
    axis tight;
    axis square;
    lims = [min([xlim,ylim],[],'all'),max([xlim,ylim],[],'all')];
    plot(lims,lims,'--k')
end

%%
figure;
hold on;
xlabel('SD (ms)');
SD_ramp = SD_ramp / T * 1e3;
SD_non = SD_non / T * 1e3;
[~,edges] = histcounts([SD_ramp,SD_non]);
histogram(SD_ramp,edges,...
    'facecolor','k');
histogram(SD_non,edges,...
    'facecolor','r');