function [batchdata, batchtargets] = getbatches(data, labels, nclasses, batchsize)

if nargin <4
    batchsize = 100;
end
% for training
[N, numdims] = size(data);
N = floor(N/batchsize)*batchsize;

if size(labels,2)==1
u= unique(labels);
if nargin <3
    nclasses = length(u);
end
targets= zeros(N, nclasses);
%Create targets: 1-of-k encodings for each discrete label
for i=1:length(u)
    targets(labels==u(i),i)=1;
end
else
    targets = labels;
    nclasses = size(labels,2);
end

%Create batches
numbatches= floor(N/batchsize);
batchdata = zeros(batchsize, numdims, numbatches);
batchtargets = zeros(batchsize, nclasses, numbatches);
groups= repmat(1:numbatches, 1, batchsize);
groups= groups(1:N);
groups = groups(randperm(N));
for i=1:numbatches
    batchdata(:,:,i)= data(groups==i,:);
    batchtargets(:,:,i)= targets(groups==i,:);
end

