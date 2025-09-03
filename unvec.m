function y = unvec(x,r,c)
y=zeros(r,c);
for i=1:c
    y(1:r,i)=x((i-1)*r+1:i*r);
end