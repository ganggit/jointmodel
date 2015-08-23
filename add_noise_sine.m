function [noiseimg] = add_noise_sine(img, freq)

if nargin <2
    freq = 12;
end
% random add lines 

eflag = true; %false; %true;


idx = randi(3, 1);


[imh, imw] = size(img);
ph = randi(imh, 1);
pw = randi(imw, 1);



noiseh = max(1, floor(imh/6));
noisew = max(1, floor(imw/6));

ph = min(max(ph,noiseh), imh - noiseh); 


if idx ==1 % sine wave
hs = min(max(1, ph - floor(noiseh/2)), imh);
ws = min(max(1, pw - floor(noisew/2)), imw);

% noise(hs:hs+floor(noiseh/2), ws:ws+floor(noisew/2)) = 1;
x = 1:0.1:imw;
y = noiseh*sin(2*pi*x/freq);
y = floor(y -min(y))+1;
y = max(min(y+ph, imh),1);
sineimg = zeros(size(img));

idx = sub2ind([imh imw], y', floor(x'));

sineimg(idx) = 1;
if eflag
    SE = [0 1 0;1 1 1;0 1 0];
    sineimg = imdilate(sineimg, SE);
end
noiseimg = img+sineimg > 0.2;

elseif idx ==2 % vetical lines    
      noiseimg = zeros(size(img));
      noiseimg(:, pw) = 1;
      if eflag
          SE = [0 1 0;1 1 1;0 1 0];
          noiseimg = imdilate(noiseimg, SE);
      end
      noiseimg = img+noiseimg > 0.2;
else   % horizontal lines
    
      noiseimg = zeros(size(img));
      noiseimg(ph,:) = 1;
      if eflag
          SE = [0 1 0;1 1 1;0 1 0];
          noiseimg = imdilate(noiseimg, SE);
      end
      noiseimg = img+noiseimg > 0.2;
end