
fname = 'test.h5';
data = readh5data(fname);

image(data('map'))


% because we save the image array as (Y, X) the image is flipped,
% see http://matplotlib.1069221.n5.nabble.com/Imshow-x-and-y-transposed-td19725.html
