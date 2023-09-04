%% initialization
warning('off');
close all;
clear;
clc;

%% notes
% beta depends on time units (s or ms), and firing rate;
% rho and beta are redundant;
% stereoptypy is a circular criterion for the point of decodability

%% TODO:
% - stereotypy
% - model comparison with polynomial regression?

%% seed fixing
% rng(0);

%% tensor settings
T = 100;    % time points
N = 30;     % neurons
K = 100;    % trials
B = 5;     % bootstrap iterations

%% time settings
ti = 1;
tf = T;
t = linspace(ti,tf,T);
dt = diff(t(1:2));

%% "ramping" criteria

% "monotonocity" criteria
rho_monotonocity_cutoff = .5;
beta_monotonocity_cutoff = .05;
pval_monotonocity_cutoff = .05;

% stereotypy criteria
rho_stereotypy_cutoff = .5;
pval_stereotypy_cutoff = .01;

%% variability parameters
gammas = exprnd(15,N,1);
lambdas = exprnd(.2*(tf-ti),N,1);
mus = linspace(ti,tf,N)';
sigmas = repmat(.1*(tf-ti),N,1);

%% smoothing kernel settings
kernel = gammakernel('peakx',5,'binwidth',dt);
t_padded = ti + kernel.paddx(1) : dt : tf + kernel.paddx(end);

%% color settings
clrs = cool(N);

%% bootstrapping

% preallocation
MU_ramp = nan(T,B);
MU_non = nan(T,B);
SD_ramp = nan(B,1);
SD_non = nan(B,1);
p_ramp = nan(B,1);

% iterate through bootstrap iterations
for bb = 1 : B
%     progressreport(bb,B,'bootstrapping')
    
    %% generate fake data
    
    % preallocation
    X = nan(T,N,K);
    R = nan(T,N,K);
    spk_times = cell(N,K);
    
    % iterate through neurons
    for nn = 1 : N
        for kk = 1 : K
            X(:,nn,kk) = generativerate(...
                t,gammas(nn),mus(nn),lambdas(nn),sigmas(nn));
            x_padded = generativerate(...
                t_padded,gammas(nn),mus(nn),lambdas(nn),sigmas(nn));
            dur = tf - ti - kernel.paddx(1) + kernel.paddx(2);
            [n,ts] = poissonprocess(x_padded,dur);
            spk_times = ts + ti + kernel.paddx(1);
            spk_counts = histcounts(spk_times,'binedges',t_padded);
            spk_counts = spk_counts / dt;
            R(:,nn,kk) = conv(spk_counts,kernel.pdf,'valid');
        end
    end
    
    %% compute cross-trial averages
    x = nanmean(X,3);
    r = nanmean(R,3);
    
    %% compute correlation coefficients
    rhos_monotonocity = nan(N,1);
    
    % iterate through neurons
    for nn = 1 : N
        rhos_monotonocity(nn) = corr(t',r(:,nn));
    end
    
    %% perform linear regression
    mdls = cell(N,1);
    pvals_monotonocity = nan(N,1);
    betas_monotonocity = nan(N,1);
    
    % iterate through neurons
    for nn = 1 : N
        mdls{nn} = fitlm(t,r(:,nn));
        p = polyfit(t,r(:,nn),1);
        pvals_monotonocity(nn) = mdls{nn}.Coefficients.pValue(end);
        betas_monotonocity(nn) = p(1); % mdls{nn}.Coefficients.Estimate(end);
    end
    
    %% stereotypy assessment
    P = 10;
    k = floor(K / P);
    shuffled_trial_idcs = randperm(K,K);
    
    % preallocation
    r_partitions = nan(T,N,P);
    rhos_stereotypy = nan(N,P);
    pvals_stereotypy = nan(N,P);
    
    % iterate through partitions
    for pp = 1 : P
        partition_idcs = shuffled_trial_idcs((1 : k) + (pp - 1) * k);
        r_partitions(:,:,pp) = nanmean(R(:,:,partition_idcs),3);
    end
    
    % compute reference
    r_ref = nanmean(r_partitions,3);
    
    % iterate through partitions
    for pp = 1 : P
        
        % iterate through neurons
        for nn = 1 : N
            [rhos,pvals] = corrcoef(r_ref(:,nn),r_partitions(:,nn,pp));
            rhos_stereotypy(nn,pp) = rhos(1,2);
            pvals_stereotypy(nn,pp) = pvals(1,2);
        end
    end
    
    %     for nn = 1 : N
    %         figure('windowstyle','docked');
    %         hold on;
    %         plot(t,r_ref(:,nn),'linewidth',2);
    %         for pp = 1 : P
    %             plot(t,r_partitions(:,nn,pp),'linewidth',1);
    %         end
    %         text(.05,.95,...
    %             sprintf('\\gamma=%.1f\n\\lambda=%.1f',...
    %             gammas(nn),lambdas(nn)),...
    %             'units','normalized')
    %     end
    
    %% neuron selection
    
    % flag "monotonic" neurons
    monotonicity_flags = ...
        abs(rhos_monotonocity) > rho_monotonocity_cutoff & ...
        abs(betas_monotonocity) > beta_monotonocity_cutoff & ...
        pvals_monotonocity <= pval_monotonocity_cutoff;
    
    % flag stereotypical neurons
    stereotypy_flags = ...
        mean(rhos_stereotypy > rho_stereotypy_cutoff,2) == 1 & ...
        mean(pvals_stereotypy < pval_stereotypy_cutoff,2) == 1;
    
    % flag "ramping" neurons
    ramp_flags = ...
        monotonicity_flags & ...
        stereotypy_flags;
    
    [mean(monotonicity_flags),mean(stereotypy_flags),mean(ramp_flags)]
    
    % store proportion of ramping neurons
    p_ramp(bb) = nanmean(ramp_flags);
    
    %% naive bayes decoder
    %     tensor_ramp = cat(3,r(:,ramp_flags),r(:,ramp_flags));
    %     tensor_non = cat(3,r(:,~ramp_flags),r(:,~ramp_flags));
    train_flags = ismember(1:K,randperm(K,round(K/2)));
    tensor_ramp = cat(3,...
        nanmean(R(:,ramp_flags,train_flags),3),...
        nanmean(R(:,ramp_flags,~train_flags),3));
    tensor_non = cat(3,...
        nanmean(R(:,~ramp_flags,train_flags),3),...
        nanmean(R(:,~ramp_flags,~train_flags),3));
    
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
    [P_tR_ramp,P_Rt(:,ramp_flags,:),~,~] = ...
        naivebayestimedecoder(tensor_ramp,opt);
    [P_tR_non,P_Rt(:,~ramp_flags,:),~,~] = ...
        naivebayestimedecoder(tensor_non,opt);
    
    % normalization
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
    MU_ramp(:,bb) = mu_ramp;
    MU_non(:,bb) = mu_non;
    SD_ramp(bb) = nanmean(sd_ramp);
    SD_non(bb) = nanmean(sd_non);
    
    %%
    if bb < B
        continue;
    end
    
    %% plot single-neuron averages
    figure('position',[40.2000 41.8000 217.6000 740.8000]);
    set(gca,...
        'xlim',[ti,tf],...
        'ycolor','none',...
        'nextplot','add');
    xlabel('time (a.u.)');
    
    % plot average firing rate
    plot(t,x+(1:N)*quantile(gammas,.85),...
        'color','k',...
        'linewidth',1);
    plot(t,r+(1:N)*quantile(gammas,.85),...
        'color','r',...
        'linewidth',1);
    
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
        'ylim',[0,max(gammas)] + [-1,1] * 0.05 * max(gammas),...
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
            'linestyle',repmat('-',1+(pvals_monotonocity(nn)>pval_monotonocity_cutoff),1),...
            'linewidth',1);
        
        % annotate correlation coefficient
        text(1.05,nn/N,sprintf('\\rho = %.2f',rhos_monotonocity(nn)),...
            'color',clrs(nn,:),...
            'horizontalalignment','left',...
            'units','normalized');
        
        % annotate regression statistics
        text(-.05,nn/N,...
            sprintf('\\beta = %.2f, p-value = %.2f',betas_monotonocity(nn),pvals_monotonocity(nn)),...
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

%%
figure(...
    'name','posterior_means',...
    'numbertitle','off',...
    'windowstyle','docked');
n_rows = 1;
n_cols = 2;
sps = gobjects(n_rows,n_cols);
for rr = 1 : n_rows
    for cc = 1 : n_cols
        sp_idx = cc + (rr - 1) * n_cols;
        sps(rr,cc) = subplot(n_rows,n_cols,sp_idx);
        xlabel(sps(rr,cc),'Real time (a.u.)');
        ylabel(sps(rr,cc),'Decoded time (a.u.)');
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

% ramping posteriors
title(sps(1),sprintf('Ramping neurons (%.0f%%)',nanmean(p_ramp)*100));
errorbar(sps(1),t,nanmean(MU_ramp,2),nanstd(MU_ramp,0,2),...
    'color','k',...
    'capsize',0);
% plot(sps(1),xlim(sps(1)),ylim(sps(1)),'--k');

% non-ramping posteriors
title(sps(2),sprintf('Non-ramping neurons (%.0f%%)',(1-nanmean(p_ramp))*100));
errorbar(sps(2),t,nanmean(MU_non,2),nanstd(MU_non,0,2),...
    'color','k',...
    'capsize',0);
% plot(sps(2),xlim(sps(2)),ylim(sps(2)),'--k');

%% firing rate function
function x = generativerate(time,gamma,mu,lambda,sigma)
x_pdf = normpdf(time,normrnd(mu,lambda),sigma);
x = gamma * x_pdf ./ max(x_pdf);
end
