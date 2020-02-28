%% Test
clear all
clc
load('IphoneCamera.mat')

filename = [cd,'/images/Image-30cm-ours.jpeg'];

%Read image and resize
img = imread(filename);
imOrig = imresize(img,[1200,1200]);
imshow(imOrig)

%undistort image
[im, newOrigin] = undistortImage(imOrig, cameraParams, 'OutputView', 'full');
im = imOrig;

%Detect image points
%[imagePoints, boardSize] = detectCheckerboardPoints(im);
%imagePoints = imagePoints + newOrigin; % adds newOrigin to every row of imagePoints


bw = rgb2gray(im);

%Create Gaussian Filter
G = fspecial('gaussian', [15 15], 2);
bw = imfilter(bw,G,'same');

bw,thresh = edge(bw,'Canny');
bw = edge(bw,'Canny',thresh*20);
%stats = [regionprops(bw); regionprops(not(bw))]

imshow(bw); 
% hold on;
% for i = 1:numel(stats)
%     rectangle('Position', stats(i).BoundingBox, ...
%     'Linewidth', 3, 'EdgeColor', 'r', 'LineStyle', '--');
% end


%Create World Points
squareSize = 25; % in millimeters
worldPoints = generateCheckerboardPoints(boardSize, squareSize);


% Compute rotation and translation of the camera.
position = estimateCameraPose(imagePoints, worldPoints, cameraParams)
distance = norm(position)





