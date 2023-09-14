function I = segment_features(image)
    I = [];
    processed_image = denoise(image);
    resized_image = imresize(processed_image, [50, 50]);
    % Separate image segments and extract features
    % Extract the 3 Regions (Digits)
    connected = bwconncomp(resized_image,4);
    rp = regionprops(connected, 'Image');

    % If only one connected region, split it into equal parts
    if connected.NumObjects == 1
        [h1,w1] = size(rp(1).Image);
        split = round(w1/3);
        seg_1 = imcrop(rp(1).Image,[0 0 split h1]);
        seg_1_resized = imresize(seg_1, [50, 50]);
        I(1,:,:) = FeatureExtraction(seg_1_resized);
        seg_2 = imcrop(rp(1).Image,[split 0 split h1]);
        seg_2_resized = imresize(seg_2, [50, 50]);
        I(2,:,:) = FeatureExtraction(seg_2_resized);
        seg_3 = imcrop(rp(1).Image,[split*2 0 split h1]);
        seg_3_resized = imresize(seg_3, [50, 50]);
        I(3,:,:) = FeatureExtraction(seg_3_resized);
    
    % If only two regions, split larger region to two equal parts
    elseif connected.NumObjects == 2
        [h1,w1] = size(rp(1).Image);
        [h2,w2] = size(rp(2).Image);
        if w1 > w2
            seg_1 = imcrop(rp(1).Image,[0 0 round(w1/2) h1]);
            seg_2 = imcrop(rp(1).Image,[round(w1/2) 0 round(w1/2) h1]);
            seg_3 = rp(2).Image;
        else
            seg_1 = rp(1).Image;
            seg_2 = imcrop(rp(2).Image,[0 0 round(w2/2) h2]);
            seg_3 = imcrop(rp(2).Image,[round(w2/2) 0 round(w2/2) h2]);
        end
        seg_1_resized = imresize(seg_1, [50, 50]);
        seg_2_resized = imresize(seg_2, [50, 50]);
        seg_3_resized = imresize(seg_3, [50, 50]);
        I(1,:,:) = FeatureExtraction(seg_1_resized);
        I(2,:,:) = FeatureExtraction(seg_2_resized);
        I(3,:,:) = FeatureExtraction(seg_3_resized);
    
    % If three regions, just compute features
    elseif connected.NumObjects == 3
        seg_1_resized = imresize(rp(1).Image, [50, 50]);
        seg_2_resized = imresize(rp(2).Image, [50, 50]);
        seg_3_resized = imresize(rp(3).Image, [50, 50]);
        I(1,:,:) = FeatureExtraction(seg_1_resized);
        I(2,:,:) = FeatureExtraction(seg_2_resized);
        I(3,:,:) = FeatureExtraction(seg_3_resized);

    % If more than three regions ignore and set dummy values
    else
        sprintf('%01d regions detected', connected.NumObjects);
        I(1,:,:) = FeatureExtraction(zeros(50, 50));
        I(2,:,:) = FeatureExtraction(zeros(50, 50));
        I(3,:,:) = FeatureExtraction(zeros(50, 50));
    end

    I = squeeze(I);
end