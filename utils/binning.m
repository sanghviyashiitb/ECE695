function [xb, yb] = binning(x,y,B)
T = size(x,1);  d = size(x,2); N = size(y,2);

Tnew = floor(T/B);
xb = zeros(Tnew,d);  yb=zeros(Tnew,N);
for t=1:Tnew
    xb(t,:) = mean(x((t-1)*B+1:t*B,:),1);
    yb(t,:) = sum(y((t-1)*B+1:t*B,:),1);
end
end

