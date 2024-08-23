function rgb = lab_to_rgb(lab)

cform = makecform('lab2srgb');
rgb = applycform(lab,cform);

% test1 = rgb(:,:,1);
% test2 = rgb(:,:,1);
% test3 = rgb(:,:,1);
% 
% test4 = lab(:,:,1);
% test5 = lab(:,:,1);
% test6 = lab(:,:,1);
% 
% Max1 = max(test1);
% Min1 = min(test1);
% Max2 = max(test4);
% Min2 = min(test4);


end