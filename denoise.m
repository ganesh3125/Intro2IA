function processed_image = denoise(I)

    % Apply grayscale
    I1 = rgb2gray(I);

    % Apply median filter
    I2 = medfilt2(I1,[5 5]);

    % Apply gaussian filter
    I3 = imgaussfilt(I2,2);

    % Binarize image
    I4 = ~im2bw(I3,graythresh(I3));
    %I4 = imbinarize(I3,graythresh(I3));

    % Apply erode
    I5 = imerode(I4, strel('disk',4));

    % Remove small regions less than 500 pixels
    I6 = bwareaopen(I5, 500);

    % Apply dilate
    I7 = imdilate(I6, strel('square',5));
    processed_image = I7;
end