function fis = genfis_kb(Xin,varargin)

if isempty(varargin)
    option = genfisOptions('GridPartition');
else
    option = varargin{1};
end

if isa(option,'fuzzy.genfis.GridPartitionOptions')
    warningState = warning('off','fuzzy:general:warnGenfis1_Deprecation');
    restoreWarning = onCleanup(@()warning(warningState));
    fis = genfis1([Xin Xout],option.NumMembershipFunctions, ...
        option.InputMembershipFunctionType, ...
        option.OutputMembershipFunctionType);
elseif isa(option,'fuzzy.genfis.SubtractiveClusteringOptions')
    warningState = warning('off','fuzzy:general:warnGenfis2_Deprecation');
    restoreWarning = onCleanup(@()warning(warningState));
    dataScale = option.DataScale;
    if strcmp(option.DataScale,'auto')
        dataScale = [];
    end
    args = {option.ClusterInfluenceRange,dataScale, ...
        [option.SquashFactor,option.AcceptRatio, ...
        option.RejectRatio,option.Verbose]};
    if ~isempty(option.CustomClusterCenters)
        args = [args option.CustomClusterCenters];
    end
    fis = genfis2(Xin,Xout,args{:});
elseif isa(option,'fuzzy.genfis.FCMClusteringOptions')
    warningState = warning('off','fuzzy:general:warnGenfis3_Deprecation');
    restoreWarning = onCleanup(@()warning(warningState));
    fis = genfis3_kb(Xin,option.FISType,option.NumClusters, ...
        [option.Exponent option.MaxNumIteration ...
        option.MinImprovement option.Verbose],1);
else
    error(message('fuzzy:general:errGenfisOptions_InvalidType'));
end
end