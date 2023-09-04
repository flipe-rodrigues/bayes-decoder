%% initialization
warning('off');
close all;
clear;
clc;

%% seed fixing
rng(0);

%% tensor settings
T = 500;    % time points
N = 15;     % neurons
K = 1e3;    % trials

%% trial settings
trial_idcs = 1 : K;

%% stimulus settings
stimulus.scaling = 3e3;
stimulus.boundary = .5;
stimulus.set = [.2, .35, .46, .54, .65, .8];
stimulus.n = numel(stimulus.set);
stimuli = sort(stimulus.set(randi(stimulus.n,K,1)))';

%% time settings
ti = -stimulus.set(1);
tf = stimulus.set(end);
t = linspace(ti,tf,T);
dt = diff(t(1:2));

%% condition settings
condition_set = 1 + [3, 1.5, 0, -1] * .05;
C = numel(condition_set);
y = sum(rand(K,1) > (0 : C-1) / C, 2);
ctrl_flags = condition_set(y) == 1;
ctrl_idcs = find(ctrl_flags);
n_ctrl_trials = numel(ctrl_idcs);

%% speed sampling
gain_mods = nan(K,1);
offset_mods = nan(K,1);
scaling_mods = nan(K,1);
mod_sig = .05;
for kk = 1 : K
    gain_mods(kk) = 1 / normrnd(condition_set(y(kk)),mod_sig);
    offset_mods(kk) = -abs(1 - normrnd(condition_set(y(kk)),mod_sig)) * 15;
    scaling_mods(kk) = normrnd(condition_set(y(kk)),mod_sig);
end

%% color settings
warm_clr = [.85,.1,.25];
cool_clr = [.15,.85,.9];
ctrl_clr = [1,1,1] * 0;
n_warm = sum(condition_set < 1);
n_cold = sum(condition_set > 1);
clrs = flipud(unique([...
    linspace(warm_clr(1),ctrl_clr(1),n_warm+1),...
    linspace(ctrl_clr(1),cool_clr(1),n_cold+1);...
    linspace(warm_clr(2),ctrl_clr(2),n_warm+1),...
    linspace(ctrl_clr(2),cool_clr(2),n_cold+1);...
    linspace(warm_clr(3),ctrl_clr(3),n_warm+1),...
    linspace(ctrl_clr(3),cool_clr(3),n_cold+1);...
    ]','rows','stable'));

%% temporal smoothing settings
peakx = 25 / 1e3;
kernel = gammakernel('peakx',peakx,'binwidth',dt);
t_padded = ti + kernel.paddx(1) : dt : tf + kernel.paddx(end) + dt;
T_padded = numel(t_padded);

%% generate fake data

% preallocation
X_padded = nan(T_padded,N,K);
spiketimes = cell(N,K);
neuron_ids = cell(N,K);

% rate settings
fr_mod = 15;
mus = linspace(0,1,N) * tf * .5;
sig0 = .035 * tf;
web = .05 * 1;

% intantiate single trial rates
for nn = 1 : N
    progressreport(nn,N,'generating rate data');
    for kk = 1 : K
        t_flags = t_padded <= stimuli(kk) + dt;
        mu = mus(nn) * scaling_mods(kk);
        sig = sig0 + mu * web;
        X_padded(:,nn,kk) = ...
            offset_mods(kk) + gain_mods(kk) * normpdf(t_padded,mu,sig)';
    end
end
X_padded = (X_padded - min(X_padded,[],[1,3]) + 1e-6) * fr_mod;
X = X_padded(t_padded >= ti & t_padded <= tf + dt,:,:);

% instantiate spike trains
for nn = 1 : N
    progressreport(nn,N,'generating spike data');
    for kk = 1 : K
        t_flags = t_padded <= stimuli(kk);
        lambda = X_padded(t_flags,nn,kk);
        dur = min(stimuli(kk),tf) - ti - kernel.paddx(1);
        [n,ts] = poissonprocess(lambda,dur);
        spiketimes{nn,kk} = ts + ti + kernel.paddx(1);
        neuron_ids{nn,kk} = repmat(nn,n,1);
    end
end

%% plot example single-trials
figure;
set(gca,...
    'xlim',[ti,tf],...
    'ycolor','none',...
    'nextplot','add',...
    'plotboxaspectratio',[10,N,1]);
xlabel('time (a.u.)');

% iterate through trials
for kk = ctrl_idcs(randperm(sum(ctrl_flags),10))
    title(sprintf('trial: %i, condition: %i',kk,y(kk)));
    plot(t,X(:,:,kk)+(1:N)*fr_mod*10,...
        'color',clrs(y(kk),:),...
        'linewidth',.1);
    drawnow;
end

%% plot condition-split averages
figure;
set(gca,...
    'xlim',[ti,tf],...
    'ycolor','none',...
    'nextplot','add',...
    'plotboxaspectratio',[10,N,1]);
xlabel('time (a.u.)');

% preallocation
x = nan(T,N,C);

% iterate through conditions
for cc = 1 : C
    
    % compute condition mean
    x(:,:,cc) = nanmean(X(:,:,y==cc),3);
    
    % plot condition mean
    plot(t,x(:,:,cc)+(1:N)*fr_mod*10,...
        'color',clrs(cc,:),...
        'linewidth',1);
end

% plot condition means
title('condition-split average activity');
drawnow;

%% convolve spike trains with smoothing kernel

% preallocation
R = nan(T,N,K);

% instantiate single trial rates
for nn = 1 : N
    progressreport(nn,N,'convolving spike trains');
    for kk = 1 : K
        t_flags = t <= stimuli(kk);
        spks = spiketimes{nn,kk};
        spk_counts = histcounts(spks,'binedges',t_padded);
        spk_counts = spk_counts / dt;
        spk_rate = conv(spk_counts,kernel.pdf,'valid');
        R(t_flags,nn,kk) = spk_rate(t_flags) - ...
            nanmean(x(:,nn,y(kk))); % !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    end
end

% offset
R = R - min(R,[],'all');

%% average spike rates within condition
figure;
set(gca,...
    'xlim',[ti,tf],...
    'ycolor','none',...
    'nextplot','add',...
    'plotboxaspectratio',[10,N,1]);
xlabel('time (a.u.)');

% preallocation
r = nan(T,N,C+1);

% iterate through conditions
for cc = 1 : C
    condition_flags = y == cc;
    condition_idcs = trial_idcs(condition_flags);
    n_flagged_trials = sum(condition_flags);
    
    % compute condition mean
    if condition_set(cc) == 1
        train_idcs = condition_idcs(...
            randperm(n_flagged_trials,round(n_flagged_trials/2)));
        test_idcs = condition_idcs(...
            ~ismember(condition_idcs,train_idcs));
        r(:,:,C+1) = nanmean(R(:,:,train_idcs),3);
    else
        test_idcs = condition_idcs;
    end
    r(:,:,cc) = nanmean(R(:,:,test_idcs),3);
    
    % plot condition mean
    plot(t,r(:,:,cc)+(1:N)*fr_mod*10,...
        'color',clrs(cc,:),...
        'linewidth',1);
end

% plot condition means
title('condition-split average activity');
drawnow;

%% naive bayes decoder
opt = struct();
opt.n_xpoints = 100;
opt.time = t;
opt.train.trial_idcs = C + 1;
opt.train.n_trials = numel(opt.train.trial_idcs);
opt.test.trial_idcs = 1 : C;
opt.test.n_trials = numel(opt.test.trial_idcs);
opt.assumepoissonmdl = true;
opt.verbose = true;

tic
[P_tR,P_Rt,pthat,neurons] = naivebayestimedecoder(r,opt);
toc

%% plot encoding models
figure;
set(gca,...
    'xlim',[t(1),t(end)],...
    'xtick',unique([0,ti,tf,stimulus.set]),...
    'xticklabel',num2cell(unique([0,ti,tf,stimulus.set]*stimulus.scaling/1e3)),...
    'xdir','normal',...
    'ydir','normal',...
    'nextplot','add',...
    'plotboxaspectratio',[1,1,1]);
xlabel('Time (a.u.)');
ylabel('Rate (a.u.)');

% iterate through units
for nn = 1 : N
    cla;
    r_bounds = neurons(nn).x_bounds;
    r_bw = neurons(nn).x_bw;
    if range(r_bounds) == 0
        continue;
    end
    ylim(r_bounds);
    title(sprintf('Neuron: %i, bw: %.2f',nn,r_bw));
    p_Rt = squeeze(P_Rt(:,nn,:));
    p_Rt(isnan(p_Rt)) = max(p_Rt(:));
    imagesc([t(1),t(end)],r_bounds,p_Rt');
    plot(t,neurons(nn).x_mu,...
        'color','w',...
        'linewidth',1);
    for ii = 1 : stimulus.n
        plot([1,1]*stimulus.set(ii),ylim,'--w');
    end
    drawnow;
    pause(.1);
end
close;

%% plot condition-split posterior averages
figure(...
    'name','condition-split posterior averages',...
    'numbertitle','off',...
    'windowstyle','docked');
n_rows = 3;
n_cols = C;
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
    'xlim',[t(1),t(end)],...
    'ylim',[t(1),t(end)],...
    'xdir','normal',...
    'ydir','normal',...
    'nextplot','add',...
    'plotboxaspectratio',[1,1,1]);
linkaxes(sps);

% iterate through conditions
clims = quantile(P_tR(:),[0,.99]);
for cc = 1 : C
    
    %
    title(sps(1,cc),sprintf('condition: %.2f',condition_set(cc)));
    p_cond = P_tR(:,:,cc);
    imagesc(sps(1,cc),[t(1),t(end)],[t(1),t(end)],p_cond',clims);
    plot(sps(1,cc),xlim(sps(1,cc)),ylim(sps(1,cc)),'-k');
    plot(sps(1,cc),xlim(sps(1,cc)),ylim(sps(1,cc)),'--w');
    
    %
    title(sps(2,cc),sprintf('condition: %.2f',condition_set(cc)));
%     p_chance = avgfun(P_tR_chance(:,:,y(opt.test.trial_idcs)==cc),3);
    p_chance = nanmean(p_cond,1);
    imagesc(sps(2,cc),[t(1),t(end)],[t(1),t(end)],p_chance',clims);
    plot(sps(2,cc),xlim(sps(2,cc)),ylim(sps(2,cc)),'-k');
    plot(sps(2,cc),xlim(sps(2,cc)),ylim(sps(2,cc)),'--w');
    
    %
    title(sps(3,cc),sprintf('condition: %.2f',condition_set(cc)));
    p_diff = p_cond - p_chance;
%     p_diff(p_diff < 0) = 0;
    imagesc(sps(3,cc),[t(1),t(end)],[t(1),t(end)],p_diff');
    plot(sps(3,cc),xlim(sps(3,cc)),ylim(sps(3,cc)),'-k');
    plot(sps(3,cc),xlim(sps(3,cc)),ylim(sps(3,cc)),'--w');
end

%% plot superimposed condition-split posterior averages

% figure initialization
figure(...
    'name','superimposed condition-split posterior averages',...
    'numbertitle','off',...
    'color','w');

% axes initialization
axes(...
    'xlim',[0,stimulus.set(end)]+[-1,1]*.05*range(stimulus.set),...
    'ylim',[0,stimulus.set(end)]+[-1,1]*.05*range(stimulus.set),...
    'xtick',sort([0,stimulus.set]),...
    'ytick',sort([0,stimulus.set]),...
    'xticklabel',num2cell(sort([0,stimulus.set])*stimulus.scaling/1e3),...
    'yticklabel',num2cell(sort([0,stimulus.set])*stimulus.scaling/1e3),...
    'fontsize',12,...
    'linewidth',2,...
    'tickdir','out',...
    'nextplot','add',...
    'layer','top',...
    'ticklength',[1,1]*.025,...
    'plotboxaspectratio',[1,1,1]);
xlabel('Real time (s)');
ylabel('Decoded time (s)');

P2 = tensor2rgb(permute(P_tR,[2,1,3]),clrs);
imagesc([t(1),t(end)],[t(1),t(end)],P2,[0,1]);

% plot categeory boundary
plot(xlim,[1,1]*.5,'--k');
plot([1,1]*stimulus.boundary,ylim,'--k');

% plot identity line
plot(xlim,ylim,'--k');

%% plot control-subtracted posterior averages
figure(...
    'name','control-subtracted posterior averages',...
    'numbertitle','off',...
    'windowstyle','docked');
sps = gobjects(C,1);
for cc = 1 : C
    sps(cc) = subplot(1,C,cc);
    xlabel(sps(cc),'real time (a.u.)');
    ylabel(sps(cc),'decoded time (a.u.)');
end
set(sps,...
    'xlim',[t(1),t(end)],...
    'ylim',[t(1),t(end)],...
    'xdir','normal',...
    'ydir','normal',...
    'nextplot','add',...
    'plotboxaspectratio',[1,1,1]);
linkaxes(sps);

% compute control mean
p_ctrl = P_tR(:,:,condition_set == 1);

% iterate through conditions
for cc = 1 : C
    title(sps(cc),sprintf('condition: %.2f - ctrl',condition_set(cc)));
    p_cond = P_tR(:,:,cc);
    p_diff = p_cond - p_ctrl;
    imagesc(sps(cc),[t(1),t(end)],[t(1),t(end)],p_diff',[-1,1]*C/T);
    plot(sps(cc),xlim(sps(cc)),ylim(sps(cc)),'-k');
    plot(sps(cc),xlim(sps(cc)),ylim(sps(cc)),'--w');
end

%% plot slices through condition-split posterior averages

% figure initialization
figure(...
    'name','slices through condition-split posterior averages',...
    'numbertitle','off',...
    'color','w');

% slice settings
slices = [0,stimulus.set(1:end-1)];
n_slices = numel(slices);

% axes initialization
sps = gobjects(n_slices,1);
for ii = 1 : n_slices
    sps(ii) = subplot(n_slices,1,ii);
end
set(sps,...
    'xlim',[ti,tf],...
    'xtick',[0,stimulus.set],...
    'ytick',[0],...
    'ylimspec','tight',...
    'xdir','normal',...
    'ydir','normal',...
    'layer','top',...
    'clipping','off',...
    'fontsize',12,...
    'linewidth',2,...
    'tickdir','out',...
    'nextplot','add',...
    'ticklength',[1,1]*.025,...
    'nextplot','add',...
    'plotboxaspectratio',[n_slices*1.75,1,1]);
set(sps(1:end-1),...
    'xtick',[],...
    'xcolor','none',...
    'ycolor','none');

% axes labels
xlabel(sps(ii),'Decoded time (a.u.)');
arrayfun(@(ax)ylabel(ax,'P(t|R)'),sps);

% iterate through conditions
[~,cond_idcs] = sort(abs(condition_set-1),'descend');
for cc = cond_idcs
    p_slice = P_tR(:,:,cc);

    % iterate through slices
    for ii = 1 : n_slices
        slice_idx = find(t >= slices(ii),1);
        
        % plot posterior slice
        plot(sps(ii),t,p_slice(slice_idx,:),...
            'color',clrs(cc,:),...
            'linewidth',1.5);
    end
end

% axes linkage
linkaxes(sps);
ylim([0,max(ylim)] + [0,-1] * .25 * max(ylim));

% iterate through slices
for ii = 1 : n_slices

    % plot real time
    slice_idx = find(t >= slices(ii),1);
    plot(sps(ii),min(xlim),0,...
        'marker','>',...
        'markersize',10,...
        'markerfacecolor','k',...
        'markeredgecolor','none');
end

% % iterate through slices
% for ii = 1 : n_slices
%     
%     % update axes
%     set(sps(ii),...
%         'ylim',[0,max(ylim(sps(ii)))] + [0,1] * .15 * max(ylim(sps(ii))),...
%         'ytick',0,...
%         'yticklabel','0');
%     
%     % plot real time
%     slice_idx = find(t >= slices(ii),1);
%     plot(sps(ii),t(slice_idx),max(ylim(sps(ii))),...
%         'marker','v',...
%         'markersize',10,...
%         'markerfacecolor','k',...
%         'markeredgecolor','none');
% end

%% plot point estimate point-dropping averages

% point estimate selection
type = 'mode';

% figure initialization
figure(...
    'name',sprintf('point estimates (%s)',type),...
    'numbertitle','off');

% axes initialization
axes(...
    'xlim',[0,stimulus.set(end)]+[-1,1]*.05*range(stimulus.set),...
    'ylim',[0,stimulus.set(end)]+[-1,1]*.05*range(stimulus.set),...
    'xtick',sort([0,stimulus.set]),...
    'ytick',sort([0,stimulus.set]),...
    'xticklabel',num2cell(sort([0,stimulus.set])*stimulus.scaling/1e3),...
    'yticklabel',num2cell(sort([0,stimulus.set])*stimulus.scaling/1e3),...
    'fontsize',12,...
    'linewidth',2,...
    'tickdir','out',...
    'nextplot','add',...
    'ticklength',[1,1]*.025,...
    'plotboxaspectratio',[1,1,1]);
xlabel('Real time (s)');
ylabel('Decoded time (s)');

% plot categeory boundary
plot(xlim,[1,1]*.5,'--k');
plot([1,1]*stimulus.boundary,ylim,'--k');

% plot identity line
plot(xlim,ylim,'--k');

% flag longest stimulus period
stimperiod_flags = t >= 0 & t < stimulus.set(end);

% iterate through conditions
for cc = 1 : C
    
    % compute measures of central tendency & dispersion
    pthat_avg = pthat.(type)(:,cc);
    
    % plot measure of central tendency
    plot(t,pthat_avg,...
        'color',clrs(cc,:),...
        'linewidth',1.5);
    
    % plot stimulus onset
    onset_idx = 1;
    plot(t(onset_idx),pthat_avg(onset_idx),...
        'color',clrs(cc,:),...
        'linewidth',1.5,...
        'marker','o',...
        'markersize',8.5,...
        'markerfacecolor','w',...
        'markeredgecolor',clrs(cc,:));
    
    % iterate through stimuli
    for ii = 1 : stimulus.n
        offset_idx = find(t >= stimulus.set(ii),1) - 1;
        
        % plot stimulus offset
        plot(t(offset_idx),pthat_avg(offset_idx),...
            'color',clrs(cc,:),...
            'linewidth',1.5,...
            'marker','o',...
            'markersize',8.5,...
            'markerfacecolor',clrs(cc,:),...
            'markeredgecolor','k');
    end
end

%%
return;
P1 = double(imread('F:\Learning Lab Dropbox\Learning Lab Team Folder\Patlab protocols\Code\MATLab\FR\Predictive coding\sofs.jpg'));
P1 = normalize01(P1,[1,2,3]);

[w,h] = size(P1,[1,2]);
I = eye(3);
for ii = 1 : 3
    w_idcs = (1:w/3) + (ii-1) * w/3;
    h_idcs = (1:h/3) + (ii-1) * h/3;
    for jj = w_idcs
        for kk = h_idcs
            P1(jj,kk,:) = I(ii,:);
        end
    end
end

P2 = rgb2xyz(P1,clrs(4,:),clrs(2,:),clrs(1,:));
figure; imshowpair(P1,P2,'montage');
hold on;

marker_clrs = [clrs(4,:);clrs(2,:);clrs(1,:)];
for ii = 1 : 3
    w_idx = (ii-1) * h/3 + h + h/6;
    h_idx = (ii-1) * w/3 + w/3/2;
    
    plot(w_idx,h_idx,...
        'marker','o',...
        'markersize',30,...
        'markerfacecolor',marker_clrs(ii,:),...
        'markeredgecolor','none',...
        'linewidth',1.5,...
        'linestyle','--');
end

figure; histogram(P1(:));
figure; histogram(P2(:));