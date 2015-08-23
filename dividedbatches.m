function [batchdata, batchtargets, testbatchdata, testbatchtargets, numclasses] = dividedbatches(trainfeats, trainlabels, testfeats, testlabels, batchsize)

% Version 1.000
%
% Code provided by Gang Chen
%
% Permission is granted for anyone to copy, use, modify, or distribute this
% program and accompanying programs and documents for any purpose, provided
% this copyright notice is retained and prominently displayed, along with
% a note saying that the original programs are available from our
% web page.
% The programs and documents are distributed without any warranty, express or
% implied.  As the programs were written for research purposes only, they have
% not been tested to the degree that would be advisable in any important
% application.  All use of these programs is entirely at the user's own risk.


%if nargin <5
%     batchsize = 100;
%end
batchsize = 100;

% for training dataset
[totnum, numdims]=size(trainfeats);
fprintf(1, 'Size of the training dataset= %5d \n', totnum);

rand('state',0); %so we know the permutation of the training data
randomorder=randperm(totnum);

numbatches=floor(totnum/batchsize);

uclasses = unique(trainlabels);
numclasses = length(uclasses);
targets= zeros(totnum, numclasses);
for i=1:length(uclasses)
    targets(trainlabels==uclasses(i),i)=1;
end

batchdata = zeros(batchsize, numdims, numbatches);
batchtargets = zeros(batchsize, numclasses, numbatches);

for b=1:numbatches
  batchdata(:,:,b) = trainfeats(randomorder(1+(b-1)*batchsize:b*batchsize), :);
  batchtargets(:,:,b) = targets(randomorder(1+(b-1)*batchsize:b*batchsize), :);
end;
% clear trainfeats trainlabels targets;
clear targets;

% for test dataset
[totnum, numdims]=size(testfeats);
fprintf(1, 'Size of the test dataset= %5d \n', totnum);

rand('state',0); %so we know the permutation of the training data
randomorder=randperm(totnum);

numbatches=floor(totnum/batchsize);
% numdims  =  size(digitdata,2);
% batchsize = 100;
testbatchdata = zeros(batchsize, numdims, numbatches);
testbatchtargets = zeros(batchsize, numclasses, numbatches);

for i=1:length(uclasses)
    targets(testlabels==uclasses(i),i)=1;
end

for b=1:numbatches
    testbatchdata(:,:,b) = testfeats(randomorder(1+(b-1)*batchsize:b*batchsize), :);
    testbatchtargets(:,:,b) = targets(randomorder(1+(b-1)*batchsize:b*batchsize), :);
end;
% clear testfeats testlabels targets;
clear targets;

%%% Reset random seeds 
rand('state',sum(100*clock)); 
randn('state',sum(100*clock)); 



