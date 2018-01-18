pfx = fullfile(vl_root,'data', 'roofs1.jpg') ; 
I = imread(pfx) ;
figure, image(I) ;
I = single(rgb2gray(I)) ;
[f,d] = vl_sift(I) ;
perm = randperm(size(f,2)) ;
sel = perm(1:50) ;
h1 = vl_plotframe(f(:,sel)) ;
h2 = vl_plotframe(f(:,sel)) ; set(h1,'color','k','linewidth',3) ; set(h2,'color','y','linewidth',2) ;
h3 = vl_plotsiftdescriptor(d(:,sel),f(:,sel)) ; set(h3,'color','g')