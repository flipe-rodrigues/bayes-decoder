%% initialization
warning('off');
close all;
clear;
clc;

%% seed fixing
rng(0);

%% tensor settings
T = 500;    % time points
N = 16;     % neurons
K = 250;    % trials
E = 10;     % epochs
C = 7;      % conditions

%% trial settings
trial_idcs = 1 : K;

%% time settings
ti = 0;
tf = 5;
t = linspace(ti,tf,T);
dt = diff(t(1:2));
dur = tf - ti;

%% epoch settings
epoch_times = linspace(ti,tf,E+1);
epoch_levels = cumsum((rand(E,N) >= .5) * 2 - 1);
epoch_mask = rand(E,1) >= .5;

%% condition settings
condition = sum(rand(K,1) > (0 : C-1) / C, 2);

%% color settings
clrs = cool(C);

%% modulation sampling
gains = normrnd(0,1,N,C);

% iterate through neurons
for nn = 1 : N
    if rand >= .5
        direction = 'ascend';
    else
        direction = 'descend';
    end
    gains(nn,:) = sort(gains(nn,:),direction);
end

%% temporal smoothing settings
peakx = 25 / 1e3;
kernel = gammakernel('peakx',peakx,'binwidth',dt);

%% sample generative rates

% preallocation
rates_gen = zeros(T,N,K);
spike_times = cell(N,K);

% rate settings
fr_mod = 2;

% intantiate single trial rates
for nn = 1 : N
    progressreport(nn,N,'generating rate data');
    
    % iterate through epochs
    for ee = 1 : E
        epoch_flags = ...
            t >= epoch_times(ee) & ...
            t < epoch_times(ee + 1);
        if sum(epoch_flags) == 0
            continue;
        end
        if rand >= .5
            starting_point = rates_gen(find(epoch_flags,1));
        else
            starting_point = epoch_levels(ee);
        end
        rates_gen(epoch_flags,nn,:) = repmat(linspace(...
            starting_point,epoch_levels(ee),sum(epoch_flags)),K,1)';
        
        % iterate through conditions
        for cc = 1 : C
            condition_flags = condition == cc;
            rates_gen(epoch_flags,nn,condition_flags) = ...
                rates_gen(epoch_flags,nn,condition_flags) .* ...
                gains(nn,cc) .^ epoch_mask(ee);
        end
    end
end

% normalization
rates_gen = (rates_gen - min(rates_gen,[],[1,3]) + 1e-6) * fr_mod;

%% sample spike times

% instantiate spike trains
for nn = 1 : N
    progressreport(nn,N,'generating spike data');
    for kk = 1 : K
        lambda = rates_gen(:,nn,kk);
        [n,ts] = poissonprocess(lambda,dur);
        spike_times{nn,kk} = ts + ti;
    end
end


%% compute average condition-split generative rates

% preallocation
rates_gen_mu = nan(T,N,C);

% iterate through conditions
for cc = 1 : C
    
    % compute condition mean
    rates_gen_mu(:,:,cc) = nanmean(rates_gen(:,:,condition==cc),3);
end

%% compute spike counts

% preallocation
counts = nan(T,N,K);

% iterate through neurons
for nn = 1 : N
    progressreport(nn,N,'computing spike counts');
    
    % iterate through trials
    for kk = 1 : K
        counts(:,nn,kk) = histcounts(spike_times{nn,kk},...
            'binedges',[t,t(end)+dt]);
    end
end

%% convolve spike trains with smoothing kernel

% preallocation
rates_hat = nan(T,N,K);

% iterate through neurons
for nn = 1 : N
    progressreport(nn,N,'convolving spike trains');
    
    % iterate through trials
    for kk = 1 : K
        rates_hat(:,nn,kk) = ...
            nanconv2(counts(:,nn,kk),kernel.pdf,1) / dt;
    end
end

% offset
rates_hat = rates_hat - min(rates_hat,[],'all');

%% compute average condition-split estimated rates

% preallocation
rates_hat_mu = nan(T,N,C);

% iterate through conditions
for cc = 1 : C
    condition_flags = condition == cc;
    condition_idcs = trial_idcs(condition_flags);
    
    % compute condition mean
    rates_hat_mu(:,:,cc) = nanmean(rates_hat(:,:,condition_idcs),3);
end

%% plot average spike rates within condition

% iterate through neurons
for nn = 1 : N
    
    % figure initialization
    figure(...
        'windowstyle','docked');
    set(gca,...
        'xlim',[ti,tf],...
        'nextplot','add',...
        'plotboxaspectratio',[2,1,1]);
    title(sprintf('neuron %i',nn));
    xlabel('time (a.u.)');
    ylabel('rate (a.u.)');
    
    % iterate through conditions
    for cc = 1 : C
        
        % plot condition mean
        plot(t,rates_hat_mu(:,nn,cc),...
            'color',clrs(cc,:),...
            'linewidth',1);
    end
    
    % iterate through epochs
    for ee = 1 : E
        epoch_flags = ...
            t >= epoch_times(ee) & ...
            t < epoch_times(ee + 1);
        xpatch = [epoch_times(ee:ee+1),fliplr(epoch_times(ee:ee+1))];
        ypatch = sort([ylim,ylim]);
        patch(xpatch,ypatch,'k',...
            'facealpha',1-.95^epoch_mask(ee),...
            'edgecolor','none');
    end
end

%% population decoder

% preallocation
performance = nan(E,C,K);

% spike integration window
spk_win = min(.5,min(diff(epoch_times)));

% iterate through epochs
for ee = 1 : E
    t_flags = ...
        t >= epoch_times(ee) & ...
        t < epoch_times(ee) + spk_win;
    
    % design matrix
    design = squeeze(sum(rates_hat(t_flags,:,:)))';
    
    % iterate through conditions
    for cc = 1 %: C
        
        % response variable
        response = condition; % == cc;
        
        % iterate through trials
        for kk = 1 : K
            progressreport(kk,K,sprintf(...
                'cross-validating (epoch %i/%i, condition %i/%i)',ee,E,cc,C));
            
            % handle leave-one-out cross-validation
            train_flags = trial_idcs ~= kk;
            train_idcs = find(train_flags);
            shuffled_idcs = train_idcs(randperm(K-1));
            
            % linear discriminant analysis
            %             mdl = fitcdiscr(design(train_flags,:),response(train_flags),...
            %                 'discrimtype','linear');
            mdl = fitcecoc(design(train_flags,:),condition(train_flags),...
                'learners','linear',...
                'coding','onevsall');
            
            % prediction with test trial
            performance(ee,cc,kk) = ...
                mdl.predict(design(kk,:)) == response(kk);
        end
    end
end

%% plot decoding performance

% bar width settings
epoch_span = 1 / 3;
barwidth = epoch_span / C;

% figure initialization
fig = figure(...
    'name','population_decoder',...
    'color',[1,1,1]*1);

% axes initialization
axes(...
    'plotboxaspectratio',[1,1,1],...
    'nextplot','add',...
    'color','none',...
    'xlim',[1,E]+[-1,1]*.05*E,...
    'xtick',1:E,...
    'xticklabel',num2cell(1:E),...
    'xticklabelrotation',90,...
    'ylim',[0,1]+[-1,1]*.05,...
    'ytick',linspace(0,1,5),...
    'yticklabel',num2cell(linspace(0,1,5)*100),...
    'xcolor','k',...
    'clipping','off',...
    'layer','bottom');
xlabel('Task epoch');
ylabel('Decoder performance (%)');

% reference lines
plot(xlim,[1,1]*1/C,'-k',...
    'linewidth',1.5);

% iterate through epochs
for ee = 1 : E
    
    % patch epoch
    xpatch = [ee,ee+1,ee+1,ee] - 1/2;
    ypatch = sort([ylim,ylim]);
    patch(xpatch,ypatch,'k',...
        'facealpha',1-.95^epoch_mask(ee),...
        'edgecolor','none');
    
    % plot epoch delimeter
    plot([1,1]*ee+1/2,ylim,...
        'color','k',...
        'linestyle','--');
    
    % iterate through conditions
    for cc = 1 : C
        condition_flags = condition == cc;
        
        %
        plot(ee,nanmean(performance(ee,1,condition_flags)),...
            'marker','o',...
            'markersize',7.5,...
            'markerfacecolor',clrs(cc,:),...
            'markeredgecolor','w',...
            'linewidth',1.5);
    end
end

%
plot(1:E,nanmean(performance(:,1,:),3),...
    'color','k',...
    'marker','o',...
    'markersize',10,...
    'markerfacecolor','k',...
    'markeredgecolor','w',...
    'linewidth',1.5);