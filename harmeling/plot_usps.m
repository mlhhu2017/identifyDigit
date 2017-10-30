function bmm = plot_usps(x,r,dims)
%PLOT_USPS plot the USPS data.
%
% inputs:
%    x       [16*16 n] matrix of n images of size [16 16]
%    r       number of rows (default==ceil(sqrt(n)))
%    dims    default is dims = [16 16]
%
% outputs:
%    bm      the image, without output it will be plotted
%
% STH * 11APR2002

if ~exist('dims','var')|isempty(dims), dims = [16 16]; end
xxx = dims(1);
yyy = dims(2);
if size(x,1)~=xxx*yyy,
  error('the patterns are not 16x16 images')
end
total = size(x,2);
if ~exist('r','var')|isempty(r),  r = ceil(sqrt(total)); end
c = ceil(total/r);
x(:,(total+1):(c*r)) = min(x(:));  % auffuellen
bm = reshape(permute(reshape(x,[xxx yyy r c]),[2 3 1 4]),[yyy*r xxx*c]);
if nargout < 1
  imagesc(bm);
  %colormap(1-gray), 
  axis off
  axis equal
else
  bmm = bm;
end

