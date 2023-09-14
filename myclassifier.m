function S = myclassifier(img)

    %This classifier is not state of the art... but should give you an idea of
    %the format we expect to make it easy to keep track of your scores. Input
    %is the image, output is a 1 x 3 vector of the three numbers in the image
    %
    %This baseline classifier tries to guess... so should score about (3^3)^-1
    %on average, approx. a 4% chance of guessing the correct answer. 
    %
    
    %S = floor(rand(1,3)*3);

    S = [];

    % load pretrained model
    load Mdl;

    % preprocess, segment and extract features from image
    features = segment_features(img);

    % use model to predict
    pred = predict(Mdl,features);

    % convert to numbers
    S(1) = str2num(string(pred(1)));
    S(2) = str2num(string(pred(2)));
    S(3) = str2num(string(pred(3)));
    
end

