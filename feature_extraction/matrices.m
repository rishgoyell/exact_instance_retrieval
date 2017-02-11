%loads images, extracts sift features, converts them to rootsift and stores them in a matrix 
cd siftDemoV4
X=[];
Y=[];
Z=[];
count=0;
catcount=0;
numfeats = 0;
myDir = '../../Dataset/';
if ~isdir(myDir)
  errorMessage = sprintf('Error: The following folder does not exist:\n%s', myDir);
  uiwait(warndlg(errorMessage));
  return;
end
folderPattern = fullfile(myDir, '*_*');
folders = dir(folderPattern);
for k = 1:length(folders)
	myFolder = strcat('../../Dataset/',folders(k).name);
	if ~isdir(myFolder)
  		errorMessage = sprintf('Error: The following folder does not exist:\n%s', myFolder);
  		uiwait(warndlg(errorMessage));
  		return;
	end
	imgFilePattern = fullfile(myFolder,'N*');
	imgFiles = dir(imgFilePattern);
	fprintf(1, 'Now reading %s\n', myFolder);
	idx = randperm(size(imgFiles,1),5);
	catcount=catcount+1
	%Z= [Z;catcount*ones(30,1)];
	for i = idx
		count=count+1;
		baseFileName = imgFiles(i).name;
		fullFileName = fullfile(myFolder, baseFileName);
		%fprintf(1, 'Now reading %s\n', fullFileName);
		%finding SIFT features for important patches in image
		[img, features, xyz] = sift(fullFileName);
		numfeats = numfeats+size(features,1);
		X = [X;features];
		Y = [Y; (count*ones(size(features,1),1))];
		% imageArray = imread(fullFileName);
		% imshow(imageArray);  % Display image.
		% drawnow; % Force display to update immediately.
	end
end

%converting to rootSIFT
X=X';
sum_val = norm(X,1);
for n = 1:128
    X(n, :) = X(n, :)./sum_val;
end
X = single(sqrt(X));

fprintf('Done');
dlmwrite('../matrices/CrtestX.txt', X);
dlmwrite('../matrices/CrtestY.txt', Y);

