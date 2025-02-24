function fismat = genfis3_kb(Xin, fistype, cluster_n, fcmoptions, numOutp)

if nargin < 4
    fcmoptions = [];    
    if nargin < 3
        cluster_n = 'auto';
        if nargin < 2
            fistype = 'sugeno';
        end
    end
end

mftype = 'gaussmf'; % hardcoded for now

%%%%%%%%%%%%%%%%%%%%%%%%%
% Parameter Checking
%%%%%%%%%%%%%%%%%%%%%%%%%


[numData,numInp] = size(Xin);

% Check cluster_n
cluster_n = convertStringsToChars(cluster_n);
if ~isscalar(cluster_n)
    if ~isequal(cluster_n, 'auto')
        error(message('fuzzy:general:errGenfis3_InvalidClusterNumber'));			
    end
else
    if cluster_n < 1
        error(message('fuzzy:general:errGenfis3_NegativeClusterNumber'));			
    end
end

% Check fcmoptions


% Check fistype
fistype = lower(convertStringsToChars(fistype));
if ~isequal(fistype, 'mamdani') && ~isequal(fistype, 'sugeno')
    error(message('fuzzy:general:errFIS_InvalidType'));
end


if isequal(cluster_n, 'auto')
    xBounds = [min(Xin); max(Xin)];
    centers = subclust(Xin,0.5,xBounds);
    cluster_n = size(centers, 1);
end


%%%%%%%%%%%%%%%%%%%%
% FCM Clustering
%%%%%%%%%%%%%%%%%%%%
[center, U] = fcm(Xin, cluster_n, fcmoptions);


%%%%%%%%%%%%%%%%%%%%%%
% Building FIS
%%%%%%%%%%%%%%%%%%%%%

% Initialize a FIS
fisName = sprintf('%s%g%g',fistype,numInp,numOutp);
if fistype == "mamdani"
    fismat = mamfis('Name',fisName);
else
    fismat = sugfis('Name',fisName);
end
fismat.DisableStructuralChecks = true;

% Loop through and add inputs
minX = min(Xin);
maxX = max(Xin);
ranges = fuzzy.internal.utility.updateRangeIfEqual([minX ; maxX]');
mfTemplate = fismf;
mfTemplate.Type = mftype;
mf = repmat(mfTemplate,[1 cluster_n]);
var = repmat(fisvar,[1 numInp]);
for i = 1:1:numInp
    inputName = ['in' num2str(i)];
    var(i).Name = inputName;
    var(i).Range = ranges(i,:);

    % Loop through and add mf's
    for j = 1:1:cluster_n
        mf(j).Name = ['in' num2str(i) 'cluster' num2str(j)];
        mf(j).Parameters = computemfparams (mftype, Xin(:,i), U(j,:)', center(j,i));
    end   
    var(i).MembershipFunctions = mf;
end
fismat.Inputs = var;

% minX = min(Xout);
% maxX = max(Xout);
% ranges = fuzzy.internal.utility.updateRangeIfEqual([minX ; maxX]');
% var = repmat(fisvar,[1 numOutp]);
% switch lower(fistype)
%     
%     case 'sugeno'
% 
%         mfS = repmat(fismf('linear',zeros(1,length(fismat.Inputs)+1)),[1 cluster_n]);
%         % Loop through and add outputs
%         for i=1:1:numOutp
%             outputName = ['out' num2str(i)];
%             var(i).Name = outputName;
%             var(i).Range = ranges(i,:);
% 
%             % Loop through and add mf's
%             for j = 1:1:cluster_n
%                 mfS(j).Name = ['out' num2str(i) 'cluster' num2str(j)];
%                 mfS(j).Parameters = computemfparams ('linear', [Xin Xout(:,i)]);
%             end
%             var(i).MembershipFunctions = mfS;
%         end
% 
%     case 'mamdani'
%         
%         % Loop through and add outputs
%         
%         for i = 1:1:numOutp
%             outputName = ['out' num2str(i)];
%             var(i).Name = outputName;
%             var(i).Range = ranges(i,:);
% 
%             % Loop through and add mf's
%             for j = 1:1:cluster_n
%                 mf(j).Name = ['out' num2str(i) 'cluster' num2str(j)];
%                 mf(j).Parameters = computemfparams (mftype, Xout(:,i), U(j,:)', center(j,numInp+i));
%             end
%             var(i).MembershipFunctions = mf;
%         end        
%         
%     % Don't need 'otherwise' case since we already have a check for FIS
%     % type.
%     
% end
% fismat.Outputs = var;

% % Create rules
% ruleList = ones(cluster_n, numInp+numOutp+2);
% for i = 2:1:cluster_n
%     ruleList(i,1:numInp+numOutp) = i;    
% end
% fismat = addRule(fismat,ruleList);
% 
% fismat = enableCheckWithoutConstruction(fismat);
 end
%% Helper functions -------------------------------------------------------
function mfparams = computemfparams(mf,x,m,c)

switch lower(mf)
    
    case 'gaussmf'
        sigma = invgaussmf4sigma (x, m, c);
        mfparams = [sigma, c];
        
    case 'linear'
        [N, dims] = size(x);
        xin = [x(:,1:dims-1) ones(N,1)];
        xout = x(:, dims);
        b = xin \ xout;
        mfparams = b';
        
    % Don't need 'otherwise' case since 'mf' is hard coded to 'gaussmf'.
end

end
