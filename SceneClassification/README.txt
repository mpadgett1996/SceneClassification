1) unzip the folder Scene_Classification
-/bag_of_words
-/cbir
-/html
-/SIFT
-/Spatial_Pyramid
-/vlfeat

2)open up MATLAB and SetPath for all folders except /html

3) open /SceneClassification/runBuildPyramid.m in MATLAB 

  if vlfeat package is not installed, vlroc() will not function
  run ../Scene_Classification/vlfeat/toolbox/vl_setup

  in runBuildPyramid.m, make sure that the variable imageBaseDir is equal to the file path '../SpatialPyramid/images/' 

4) run runBuildPyramid.m

4.5) To run with random decision forest, use runBuildPyramidTree

Building the pyramids will generate lots of warnings!
To disable them in later runs, use: warning('off', 'last');

5) html/index.html- home page for our report. 



