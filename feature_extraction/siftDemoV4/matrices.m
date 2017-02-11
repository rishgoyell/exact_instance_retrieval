X=[];
Y=[];
count=0;
myDir = '/home/rishabh/Documents/VisRec/assignment1/Dataset/';
if ~isdir(myDir)
  errorMessage = sprintf('Error: The following folder does not exist:\n%s', myDir);
  uiwait(warndlg(errorMessage));
  return;
end
folderPattern = fullfile(myDir, '*_*');
folders = dir(folderPattern);
for k = 1:length(folders)
	myFolder = strcat('/home/rishabh/Documents/VisRec/assignment1/Dataset/',folders(k).name);
	if ~isdir(myFolder)
  		errorMessage = sprintf('Error: The following folder does not exist:\n%s', myFolder);
  		uiwait(warndlg(errorMessage));
  		return;
	end
	imgFilePattern = fullfile(myFolder,'N1*');
	imgFiles = dir(imgFilePattern);
	idx = randperm(size(imgFiles,1),10)
	for i = idx
		count=count+1;
		baseFileName = imgFiles(i).name;
		fullFileName = fullfile(myFolder, baseFileName);
		fprintf(1, 'Now reading %s\n', fullFileName);
		[img, features, xyz] = sift(fullFileName);
		X = [X;features];
		Y = [Y; (count*ones(size(features,1),1))];
		% imageArray = imread(fullFileName);
		% imshow(imageArray);  % Display image.
		% drawnow; % Force display to update immediately.
	end
end