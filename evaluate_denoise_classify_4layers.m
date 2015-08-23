function evaluate_denoise_classify_4layers

fpath = '/panfs/panfs.ccr.buffalo.edu/scratch/GangChen_CVPR/RBMLIB/Autoencoder_Code/';

batchsize =100;
numclass =10;
numfolder=5;
dataid = 'usps';
imh = 20;imw = 20;
try
    %load mnist_noise2_data.mat
    load trainpairs_noise2.mat;
    
    if(dataid =='usps')
        % change it
        imh = 20; imw =16;
    Y = Y-min(Y) +1;
    numdata = size(X,1);
    % into vector
    numclasses = max(Y);

    indx = sub2ind([numdata, numclasses], [1: numdata]', Y);
    gt = zeros(numdata, numclasses);
    gt(indx) =1;
    Y = gt;
    load('usps_split3.mat');
    %testidx = idlist(1:floor(numdata/numfolder));
    %trainidx = setdiff(idlist, testidx);
    % save('usps_split3.mat', 'testidx', 'trainidx');
    trainX = X(trainidx,:);
    trainnoiseX = noiseX(trainidx,:);
    trainY = Y(trainidx,:);

    testX = X(testidx,:);
    testnoiseX = noiseX(testidx,:);
    testY = Y(testidx,:);

    batchsize =100;
    end
catch
fname = 'MNIST.mat';
load(fullfile(fpath, fname));

[batchsize,numdims, numcases] = size(batchdata);
batchdata = permute(batchdata, [1 3 2]);
X = reshape(batchdata, batchsize*numcases, numdims); clear batchdata;

[batchsize,numdims, numcases] = size(batchtargets);
batchtargets = permute(batchtargets, [1 3 2]);
Y = reshape(batchtargets, batchsize*numcases, numdims); clear batchtargets;


[batchsize,numdims, numcases] = size(testbatchdata);
testbatchdata = permute(testbatchdata, [1 3 2]);
testX = reshape(testbatchdata, batchsize*numcases, numdims); clear testbatchdata;
[batchsize,numdims, numcases] = size(testbatchtargets);

testbatchtargets = permute(testbatchtargets, [1 3 2]);
testY = reshape(testbatchtargets, batchsize*numcases, numdims); clear testbatchtargets;


imh = 28; imw = 28;
noisetestX = testX;
for i =1:size(testX,1)
    temp = reshape(testX(i,:), imh,imw);
    temp = temp';
    testX(i,:) = reshape(temp, 1, imh*imw);
    noisetestX(i,:) = reshape(add_noise(temp), [1, imh*imw]);
end
end

% do the denosing first

%load mnist_n2_weights.mat;
load usps_noise2_weights.mat;
expectout = testX;

numclass = size(Y,2);


[testnoisebatchdata, testbatchtargets, testbatchdata] = genbatch(testnoiseX, testY, batchsize, testX);
psnrval = [];
err_cr=0;
[testnumcases testnumdims testnumbatches]=size(testnoisebatchdata);
N=testnumcases;
err=0;
testcounter = 0;
for batch = 1:testnumbatches
  data = [testnoisebatchdata(:,:,batch)];
  data = [data ones(N,1)];
  target = [testbatchtargets(:,:,batch)];
  w1probs = 1./(1 + exp(-data*w1)); w1probs = [w1probs  ones(N,1)];
  w2probs = 1./(1 + exp(-w1probs*w2)); w2probs = [w2probs ones(N,1)];
  w3probs = 1./(1 + exp(-w2probs*w3)); w3probs = [w3probs  ones(N,1)];
  w4probs = w3probs*w4; w4probs = [w4probs  ones(N,1)];
  w5probs = 1./(1 + exp(-w4probs*w5)); w5probs = [w5probs  ones(N,1)];
  w6probs = 1./(1 + exp(-w5probs*w6)); w6probs = [w6probs  ones(N,1)];
  w7probs = 1./(1 + exp(-w6probs*w7)); w7probs = [w7probs  ones(N,1)];
  dataout = 1./(1 + exp(-w7probs*w8));
  bpsnr = cal_psnr(dataout,testbatchdata(:,:,batch), imh,imw);
  psnrval = [psnrval, bpsnr];
%   err = err +  1/N*sum(sum( (data(:,1:end-1)-dataout).^2 ));
% classification
  w9probs = 1./(1 + exp(-w4probs*w9)); w9probs = [w9probs  ones(N,1)];
  w10probs = 1./(1 + exp(-w9probs*w10)); w10probs = [w10probs ones(N,1)];
  w11probs = 1./(1 + exp(-w10probs*w11)); w11probs = [w11probs  ones(N,1)];
  targetout = exp(w11probs*w_class);
  targetout = targetout./repmat(sum(targetout,2),1,numclass);

  [I J]=max(targetout,[],2);
  [I1 J1]=max(target,[],2);
  testcounter=testcounter+length(find(J==J1));
  err = err- sum(sum( target(:,1:end).*log(targetout))) ;
 end
 test_rec=err/testnumbatches;
 test_err=1- testcounter/(testnumbatches*batchsize);
 sum(psnrval)/(testnumbatches*batchsize);
 
 
 
 
 function avgmean = cal_psnr(dataout,data, imh,imw)
 avgmean =[];
 [numdata, numdims] = size(dataout);
 for i=1: numdata
 imrec = reshape(dataout(i,:), imh,imw);
  imrec =imrec./max(imrec(:));
  im1 = reshape(data(i,:), imh,imw);
  psnrval = evaluatepsnr(im1, imrec, 1);
  
   avgmean = [avgmean, psnrval];
 end

