I = zeros(400,400);
I(100:300, 100:300) = 1.0;
I = single(I);
imshow(I,[]);

x = 100;
y = 100;
scale = 5;
ang = 0;
% Specify (x;y;scale,angle) of a feature (frame) to extract 
fc = [x;y;scale;ang];
[f,d] = vl_sift(I,'frames',fc);

%% Plot it
h = vl_plotsiftdescriptor(d,f);
set(h,'color','g');
disp(f);    % x,y,scale,angle
figure, plot(d);

%% Show the image at that scale
g = fspecial('gaussian', 6*scale, scale); 
Is = imfilter(I,g);
figure, imshow(Is,[]);
[gx,gy] = gradient(Is);
x = 1:size(I,2);
y = 1:size(I,1);
hold on
quiver(x, y, gx, gy);
h = vl_plotsiftdescriptor(d,f); set(h,'color','g');