# SceneClassification

find the index.html within the html folder for a more thorough explanation.

In this project we were trying to find the "best" classifer for scene classification. 
We implemented a Spatial Pyramid to classify scenes, and improve performance on a bag-of-words representation.
A spatial pyramid works by dividing the given image into increasingy smalled sub-regions and computing histograms of the 
local features found inside each sub-region. Spatial pyramid is more efficient than bag-of-words implementaion because 
it not only detects what the objects are, but also where in the image they are. We also implemented a random decision 
forest classifer, which  We compared our spatial pyramid results and random decision forest results to the starter codes 
histogram intersection classifier to find which had the best performance.
	
 
