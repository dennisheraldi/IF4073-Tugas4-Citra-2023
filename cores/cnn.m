function label = cnn(img)
    addpath cores/pretrained-yolo-v4/
    addpath cores/pretrained-yolo-v4/src/
    addpath cores/pretrained-yolo-v4/models/
    modelName = 'YOLOv4-coco';
    model = helper.downloadPretrainedYOLOv4(modelName);
    net = model.net;
    
    % Read test image.
    image = img;
    
    % Get classnames of COCO dataset.
    classNames = helper.getCOCOClassNames;
    
    % Get anchors used in training of the pretrained model.
    anchors = helper.getAnchors(modelName);
    
    % Detect objects in test image.
    executionEnvironment = 'auto';
    [~, scores, labels] = detectYOLOv4(net, image, anchors, classNames, executionEnvironment);
    
    label_score = [string(labels) string(scores)];
    
    % Kelas valid
    valid_classes = ["bus", "car", "truck", "motorbike"];
    
    % Filter untuk hanya kelas yang valid
    valid_scores = [];
    for i = 1:size(label_score, 1)
        if any(valid_classes == label_score{i, 1})
            valid_scores = [valid_scores; label_score(i, :)];
        end
    end
    
    % Inisialisasi variabel untuk menyimpan kelas dengan skor tertinggi
    max_score = -1;
    max_class = "car"; % Default
    
    % Cari kelas dengan skor tertinggi
    for i = 1:size(valid_scores, 1)
        current_score = str2double(valid_scores{i, 2});
        if current_score > max_score
            max_score = current_score;
            max_class = valid_scores{i, 1};
        end
    end
    
    % Output kelas dengan skor tertinggi atau default
    label = max_class;
end


