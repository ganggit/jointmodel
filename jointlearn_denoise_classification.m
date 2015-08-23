function mnist_learn_denoise()

addpath('/gpfs/scratch/gangchen/GangChen_CVPR/RBMLIB/Autoencoder_Code');

dataid = 'mnist_noise1';
fpath = '/gpfs/scratch/gangchen/GangChen_CVPR/RBMLIB/';
fname = 'MNIST.mat';
lambda =1;
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

%nums = 4000;
%X = X(1:nums,:);
%Y = Y(1:nums,:);

imh = 28; imw = 28;
noiseX = X;
for i =1:size(X,1)
    temp = reshape(X(i,:), imh,imw);
    temp = temp';
    X(i,:) = reshape(temp, 1, imh*imw);
    noiseX(i,:) = reshape(add_noise_sine(temp), [1, imh*imw]);
end


idlist = randperm(size(X,1));
X = X(idlist,:);
noiseX = noiseX(idlist,:);
Y = Y(idlist,:);


testnoiseX = testX;
for i =1:size(testX,1)
    temp = reshape(testX(i,:), imh,imw);
    temp = temp';
    testX(i,:) = reshape(temp, 1, imh*imw);
    testnoiseX(i,:) = reshape(add_noise_sine(temp), [1, imh*imw]);
end

% save('mnist_noise1.mat', 'X', 'noiseX', 'Y','testX','testnoiseX', 'testY');

batchsize =100;
jointlearn_dae_4layers(X, noiseX, Y,testX,testnoiseX, testY, batchsize, lambda, dataid);


end