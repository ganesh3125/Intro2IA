clear all;

data = importdata('Train/labels.txt');
img_nrs = data(:,1);
true_labels = data(:,(2:4));

my_labels = zeros(size(true_labels));
train_size = 1000;
test_size = 200;

train_data = zeros(train_size*3,2500);
validation_data = zeros(test_size*3,2500);
train_data = [];
validation_data = [];

train_labels_1 = true_labels(1:train_size,:); 
validation_labels_1 = true_labels(train_size+1:train_size+test_size,:); 

for n = 1:train_size
    k = img_nrs(n);
    im = imread(sprintf('Train/captcha_%04d.png', k));

%     r_im = imresize(im, 0.2);
%     %new
% 
%     cap1_bw = rgb2gray(r_im);
%     %enhance_f = imadjust(cap1_bw);
%     enhance_f = adapthisteq(cap1_bw);
%     
%     
%     %Structuring element
%     se1 = strel('disk', 5);
%     se2 = strel('ball', 5, 5);
%     
%     de = imdilate(enhance_f, se2);
%     er = imerode(de, se1);
%     
%     %Iblur1 = wiener2(er,[3 3]);
%     Iblur1 = imgaussfilt(er,5);
    
%     train_data(k,:,:) = Iblur1 < 170;
    features = segment_features(im);
    train_data(end+1,:) = features(1,:);
    train_data(end+1,:) = features(2,:);
    train_data(end+1,:) = features(3,:);

    %end new
    %train_data(k,:,:) = rgb2gray(r_im);
end

for n = 1:test_size
    k = img_nrs(n+train_size);
    im = imread(sprintf('Train/captcha_%04d.png', k));
    %validation_data(k,:,:) = segment_features(im);

    features = segment_features(im);
    validation_data(end+1,:) = features(1,:);
    validation_data(end+1,:) = features(2,:);
    validation_data(end+1,:) = features(3,:);
%     validation_data(k,:,:) = rgb2gray(r_im);
end

train_labels_2 = [];
for n = 1:train_size
    train_labels_2(end+1) = string(train_labels_1(n,1));
    train_labels_2(end+1) = string(train_labels_1(n,2));
    train_labels_2(end+1) = string(train_labels_1(n,3));
end

validation_labels_2 = [];
for n = 1:test_size
    validation_labels_2(end+1) = string(validation_labels_1(n,1));
    validation_labels_2(end+1) = string(validation_labels_1(n,2));
    validation_labels_2(end+1) = string(validation_labels_1(n,3));
end

train_labels = categorical(transpose(train_labels_2));
validation_labels = categorical(transpose(validation_labels_2));

%save("captcha_data.mat", "train_data","train_labels","validation_data","validation_labels");


fprintf('Building model...\n');
t=tic;

% kNN classifier
k=3;
Mdl = fitcknn(train_data,train_labels, 'NumNeighbors',k, 'BreakTies','nearest');
toc(t)


% Ensemble of decision trees classifier (AdaBoost)
%tr = templateTree('MaxNumSplits',700); % Do evaluate different number of splits!
%Mdl = fitcensemble(train_patterns,train_labels, 'Learners',tr) % AdaBoost i the default Multiclass method

% SVM classifier
%tr = templateSVM('KernelFunction','linear');
%Mdl = fitcecoc(train_patterns,train_labels, 'Learners',tr) % ECOC model for Multiclass problems

save Mdl

fprintf('\nResubstitution error: %5.2f%%\n\n',100*resubLoss(Mdl));

% If ensemble, then view the first descision tree; click on the nodes to display data about them
if isa(Mdl,'classreg.learning.classif.ClassificationEnsemble')
	view(Mdl.Trained{1},'Mode','graph');
end


fprintf('Predicting validation set...\n');
t=tic;
validation_pred = predict(Mdl,validation_data);
toc(t);



accuracy = mean(validation_pred == validation_labels);
fprintf('Validation accuracy: %5.2f%%\n',accuracy*100);

f=figure(2);
if (f.Position(3)<800)
	set(f,'Position',get(f,'Position').*[1,1,1.5,1.5]); %Enlarge figure
end
confusionchart(validation_labels, validation_pred, 'ColumnSummary','column-normalized', 'RowSummary','row-normalized');
title(sprintf('Validation accuracy: %5.2f%%\n',accuracy*100));
