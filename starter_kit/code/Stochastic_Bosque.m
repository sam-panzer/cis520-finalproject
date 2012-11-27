function Random_Forest = Stochastic_Bosque(Data,Labels,varargin)

%Random_Forest = Stochastic_Bosque(Data,Labels,varargin)
%
%   Creates an ensemble of CARTrees using Data(samplesXfeatures).
%   The following parameters can be set :
%
%       ntrees       : number of trees in the ensemble (default 50)
%
%       oobe         : out-of-bag error calculation, 
%                      values ('y'/'n' -> yes/no) (default 'n')
%
%       nsamtosample : number of randomly selected (with
%                      replacement) samples to use to grow
%                      each tree (default num_samples)
%
%
%   Furthermore the following parameters can be set regarding the
%   trees themselves :
%
%       method       : the criterion used for splitting the nodes
%                           'g' : gini impurity index (classification)
%                           'c' : information gain (classification)
%                           'r' : squared error (regression)
%
%       minparent    : the minimum amount of samples in an impure node
%                      for it to be considered for splitting
%
%       minleaf      : the minimum amount of samples in a leaf
%
%       weights      : a vector of values which weigh the samples 
%                      when considering a split
%
%       nvartosample : the number of (randomly selected) variables 
%                      to consider at each node 

okargs =   {'minparent' 'minleaf' 'nvartosample' 'ntrees' 'nsamtosample' 'method' 'oobe' 'weights' 'depth'};
defaults = {2 1 round(sqrt(size(Data,2))) 50 numel(Labels) 'c' 'n' [] 3};
[eid,emsg,minparent,minleaf,m,nTrees,n,method,oobe,W,D] = getargs(okargs,defaults,varargin{:});
%minparent = 2;
%minleaf = 1;
%m = round(sqrt(size(Data,2)));
%nTrees = 50;
%n = numel(Labels) ;
%method = 'c';
%oobe = 'n';
%W = [];

t = CTimeleft(nTrees);
for i = 1 : nTrees
    t.timeleft();
    TDindx = round(numel(Labels)*rand(n,1)+.5);
%    TDindx = unique(TDindx);
%     TDindx = randsample(numel(Labels),n,true
%     TDindx = unique(TDindx);
    
    Random_ForestT = dt_train_multi(Data(TDindx,:),Labels(TDindx),D);
        %'minparent',minparent,'minleaf',minleaf,'nvartosample',m,'weights',W);
    
    Random_ForestT.method = method;

    Random_ForestT.oobe = 1;
    if strcmpi(oobe,'y')
        NTD = setdiff(1:numel(Labels),TDindx);
        tree_output = dt_value(Random_ForestT,Data(NTD,:))';
        
        switch lower(method)        
            case {'c','g'}                
                Random_ForestT.oobe = numel(find(tree_output-Labels(NTD)==0))/numel(NTD);
            case 'r'
                Random_ForestT.oobe = sum((tree_output-Label(NTD)).^2);
        end        
    end
    
    Random_Forest(i) = Random_ForestT;
end