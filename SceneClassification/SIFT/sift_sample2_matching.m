pfx = fullfile(vl_root,'data', 'roofs1.jpg') ; I = imread(pfx) ;
figure; image(I) ;
Ia = single(rgb2gray(I)) ;
pfx = fullfile(vl_root,'data', 'roofs2.jpg') ; I = imread(pfx) ;
figure, image(I) ;
Ib = single(rgb2gray(I)) ;
[fa, da] = vl_sift(Ia) ;
[fb, db] = vl_sift(Ib) ;
[matches, scores] = vl_ubcmatch(da, db) ;

m1= fa (1:2,matches(1,:)); m2=fb(1:2,matches(2,:));
m2(1,:)= m2(1,:)+size(Ia,2)*ones(1,size(m2,2)); X=[m1(1,:);m2(1,:)];
Y=[m1(2,:);m2(2,:)];
c=[Ia Ib];
figure, imshow(c,[]);
hold on;
line(X(:,1:3:100),Y(:,1:3:100))