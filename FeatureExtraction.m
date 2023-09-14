function F=FeatureExtraction(I)
% Compute a row-vector of feature values for an image I

	%F=[]; % Empty feature vector to add stuff to

	%F=[F,transpose(I(:))]; % Simplest possible feature extraction, just return the image intensity values as a vector
    %F = [F, detectSIFTFeatures(I)];
    %F = [F, hu_moments(I(:))];
%     T = graythresh(I<150);
%     binary = imbinarize(I,T);

    % Use flattened pixel intensities array as a feature
    F = reshape(I.',1,[]);
    %F = ShapeFeats(I);
end
