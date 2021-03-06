%% Test
clear all
clc
load('IphoneCamera.mat')

npics = 1;
plotFlag = true;

if(plotFlag)
   figure; hold on; grid on;
    plot(0,0,'o')
    axis([-100 100 -200 10]);
end
tic
for i = 1:npics
    i/npics*100
    
    %filename = [cd,'/images/Circle/Image-' num2str(i,'%05d') '.jpeg'];
    filename = [cd,'/images/Image-30cm.jpeg'];

    img = imread(filename);
    imOrig = imresize(img,[1200,1200]);
%        ims{i} = imOrig;

    [im, newOrigin] = undistortImage(imOrig, cameraParams, 'OutputView', 'full');
    im = imOrig;

    [imagePoints, boardSize] = detectCheckerboardPoints(im);

    squareSize = 25; % in millimeters
    worldPoints = generateCheckerboardPoints(boardSize, squareSize);

    imagePoints = imagePoints + newOrigin; % adds newOrigin to every row of imagePoints

    % Compute rotation and translation of the camera.
    [R, t] = extrinsics(imagePoints, worldPoints, cameraParams);
    ts(i,:) = t;
    
    % Compute camera pose.
    [orientation, location] = extrinsicsToCameraPose(R, t);

    positions(i,1) = location(1)/10;
    positions(i,2) = location(3)/10;
    
    if (plotFlag)
        scatter(positions(i,1),positions(i,2),'r','x');
        %pause(1e-3)
        drawnow()
    end
    
end
toc

