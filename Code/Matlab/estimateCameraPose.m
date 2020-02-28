function o = estimateCameraPose(imagePoints, worldPoints, cameraParams)

[R, t] = extrinsics(imagePoints, worldPoints, cameraParams);
[orientation, location] = extrinsicsToCameraPose(R, t);

o = location;

end