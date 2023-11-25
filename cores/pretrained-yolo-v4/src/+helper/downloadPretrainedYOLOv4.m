function model = downloadPretrainedYOLOv4(modelName)
% The downloadPretrainedYOLOv4 function downloads a YOLO v4 network 
% pretrained on COCO dataset.
%
% Copyright 2021 The MathWorks, Inc.

supportedNetworks = ["YOLOv4-coco", "YOLOv4-tiny-coco"];
validatestring(modelName, supportedNetworks);

dataPath = 'cores/pretrained-yolo-v4/models';
netMatFileFullPath = fullfile(dataPath, [modelName, '.mat']);
netZipFileFullPath = fullfile(dataPath, [modelName, '.zip']);

if ~exist(netMatFileFullPath,'file')
    if ~exist(netZipFileFullPath,'file')
        fprintf(['Downloading pretrained ', modelName ,' network.\n']);
        fprintf('This can take several minutes to download...\n');
        url = ['https://ssd.mathworks.com/supportfiles/vision/deeplearning/models/yolov4/', modelName, '.zip'];
        websave(netZipFileFullPath, url);
        fprintf('Done.\n\n');
        unzip(netZipFileFullPath, dataPath);
        % Hapus file zip setelah di-unzip
        delete(netZipFileFullPath);
    else
        fprintf(['Pretrained ', modelName, ' network already exists.\n\n']);
        unzip(netZipFileFullPath, dataPath);
        % Hapus file zip setelah di-unzip
        delete(netZipFileFullPath);
    end
else
    fprintf(['Pretrained ', modelName, ' network already exists.\n\n']);
end

model = load(netMatFileFullPath);
end
