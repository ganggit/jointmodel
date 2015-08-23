function noiseimg = add_noise(img, fw)

if nargin <2
    fw =6; % default value
end

[imh, imw] = size(img);
noise = zeros(imh, imw);
ph = randi(imh, 1);
pw = randi(imw, 1);

noiseh = max(1, floor(imh/fw));
noisew = max(1, floor(imw/fw));

hs = min(max(1, ph - floor(noiseh/2)), imh);
ws = min(max(1, pw - floor(noisew/2)), imw);

% noise(hs:hs+floor(noiseh/2), ws:ws+floor(noisew/2)) = 1;
noise(hs:hs+floor(noiseh/2), :) = 1;
noise(:, ws:ws+floor(noisew/2)) = 1;
angle = randi(360, 1);
out = imrotate(noise,angle,'bicubic', 'crop');
noiseimg = (double(img) + out) >0;

end