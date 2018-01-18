I1 = imread('cameraman.tif');
I1 = single(I1); % Convert to single precision floating point
imshow(I1,[]);
[f1,d1] = vl_sift(I1);

% Find the feature closest to the center of the image
dx = size(I1,2)/2 - f1(1,:); 
dy = size(I1,1)/2 - f1(2,:); 
distsq = dx.^2 + dy.^2; 
[~,i1] = min(distsq);

% Show the SIFT feature
h = vl_plotframe(f1(:,i1)) ; set(h,'color','y','linewidth',2) ;


%% Resize by a factor of 2
I2 = imresize(I1, 2);       
I2 = imrotate(I2, 30);
figure, imshow(I2,[]);
[f2,d2] = vl_sift(I2);

% Find the feature closest to the center of the image
dx = size(I2,2)/2 - f2(1,:);
dy = size(I2,1)/2 - f2(2,:);
distsq = dx.^2 + dy.^2;
[~,i2] = min(distsq);
% Show the SIFT feature
h = vl_plotframe(f2(:,i2)) ; set(h,'color','y','linewidth',2) ;
disp(f1(:,i1));     % Print (x,y,scale,ang)
disp(f2(:,i2));     % Print (x,y,scale,ang)

%Plot and overlay descriptors
figure, plot(d1(:,i1), 'r');
hold on, plot(d2(:,i2), 'g');