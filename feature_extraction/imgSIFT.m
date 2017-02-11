%loads images, extracts sift features, converts them to rootsift and stores them in a matrix 
cd siftDemoV4
X=[];
Y=[];
count=0;
numfeats = 0;
myDir = '../SampleTestSet2/';	%make this point to appropriate location
if ~isdir(myDir)
  errorMessage = sprintf('Error: The following folder does not exist:\n%s', myDir);
  uiwait(warndlg(errorMessage));
  return;
end
% folderPattern = fullfile(myDir, '*_*');
% folders = dir(folderPattern);
% for k = 1:length(folders)
	% myFolder = strcat('../../Dataset/',folders(k).name);
	% if ~isdir(myFolder)
 %  		errorMessage = sprintf('Error: The following folder does not exist:\n%s', myFolder);
 %  		uiwait(warndlg(errorMessage));
 %  		return;
	% end
imgFilePattern = fullfile(myDir,'*.jpg');
imgFiles = dir(imgFilePattern);
for i = 1:length(imgFiles)
	count=count+1;
	baseFileName = imgFiles(i).name;
	fullFileName = fullfile(myDir, baseFileName);
	fprintf(1, 'Now reading %s\n', fullFileName);
	%finding SIFT features for important patches in image
	[img, features, xyz] = sift(fullFileName);
	numfeats = numfeats+size(features,1);
	X = [X;features];
	Y = [Y; (count*ones(size(features,1),1))];
	imageArray = imread(fullFileName);
	imshow(imageArray);  % Display image.
	drawnow; % Force display to update immediately.
end


%converting to rootSIFT
X=X';
sum_val = norm(X,1);
for n = 1:128
    X(n, :) = X(n, :)./sum_val;
end
X = single(sqrt(X));
X=X';
fprintf('Done');
dlmwrite('tempXe.txt', X);
dlmwrite('tempYe.txt', Y);